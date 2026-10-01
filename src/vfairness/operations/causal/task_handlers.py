"""
Task handlers for the task consumer (a task-queue worker process).

The consumer routes incoming `vfairness_causal_*` task types to the
corresponding Python op via
this module. Each handler:
    - reads the request payload (serialized DAG + dataset_ref + options)
    - loads the dataset (either passed inline as a CSV/JSON sample or fetched
      from the assessment storage when storageKey is given)
    - calls the matching vfairness.operations.causal op
    - returns a `{success, data}` envelope ready for submit-result

The consumer is responsible for the data-loading step. We expose two helpers
(`load_dataset_from_payload`, `_normalize_options`) that codify the contract.

To register handlers in the consumer (Node-side), call this module via the
existing python-bridge subprocess pattern:

    {
      "task_type": "vfairness_causal_mediate",
      "module": "vfairness.operations.causal.task_handlers",
      "function": "handle_mediate"
    }

(Adapt to the consumer's actual handler-registration shape.)
"""

from __future__ import annotations

import io
import json
import numbers
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from ..._triage import is_measured
from .attribute import attribute_distribution_change
from .counterfactual import compute_counterfactual
from .identify import identify_paths
from .mediate import decompose_mediation
from .refute import run_refutation_suite

# Payload helpers


def load_dataset_from_payload(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Resolve the dataset for a causal task.

    Supported shapes inside `payload['dataset_ref']`:
        - {"inline_csv": "<csv text>"}
        - {"inline_json": [<row dicts>]}
        - {"storage_key": "...", "assessment_id": "..."}  -- the consumer is
          expected to fetch the file from Supabase Storage, decrypt with the
          assessment key, and put the resulting bytes/text in
          payload["dataset_bytes"] before calling the handler.

    Returns None if no dataset is available OR if what was supplied could not be
    read as a dataset; data-bound ops then return a helpful error. Every None
    carries a ``UserWarning`` naming which of those it is.

    ``dataset_bytes`` is decoded STRICTLY as UTF-8, whether it arrives as
    ``bytes``, a ``bytearray`` or a ``memoryview`` (a decrypt or buffer step
    produces the latter two, and until BGL6 they bypassed the strict decode
    entirely). Pass ``payload["dataset_encoding"]`` (for example ``"latin-1"`` or
    ``"cp1252"``) to decode an export in another encoding deliberately; see the
    measurement in :func:`_decode_dataset_bytes`.
    """
    ref = payload.get("dataset_ref") or {}

    if payload.get("dataset_bytes"):
        # BGL5 AUDIT, 2026-09-27. Supplying BOTH was resolved silently in favour
        # of dataset_bytes, so half of an ambiguous request disappeared without a
        # word. Measured: inline_csv dropped, warnings []. It still loses (the
        # pre-loaded file is the more specific instruction), but it says so.
        if ref.get("inline_csv") or ref.get("inline_json"):
            warnings.warn(
                "load_dataset_from_payload: the payload carries BOTH pre-loaded "
                "dataset_bytes and an inline dataset_ref. The pre-loaded bytes were "
                "used and the inline data was IGNORED; send one or the other so the "
                "dataset the op ran on is not decided by this function's precedence.",
                UserWarning,
                stacklevel=2,
            )
        raw = _decode_dataset_bytes(payload["dataset_bytes"], payload.get("dataset_encoding"))
        if raw is None:
            return None
        return _parse_csv_text(raw, "dataset_bytes")

    inline_csv = ref.get("inline_csv")
    if inline_csv:
        return _parse_csv_text(str(inline_csv), "dataset_ref.inline_csv")

    inline_json = ref.get("inline_json")
    if inline_json:
        return pd.DataFrame(inline_json)

    if ref.get("storage_key"):
        # A DIFFERENT None from "nothing was supplied": the caller named a stored
        # file and the consumer was supposed to fetch, decrypt and attach it as
        # dataset_bytes before calling this. Saying so distinguishes a consumer
        # step that did not run from a request that carried no dataset.
        warnings.warn(
            f"load_dataset_from_payload: dataset_ref names storage_key "
            f"{ref.get('storage_key')!r} but no dataset_bytes were attached, so nothing "
            f"was loaded. The consumer fetches and decrypts the stored file and puts it "
            f"in payload['dataset_bytes'] before calling a handler; that step did not "
            f"run. This is a could-not-load, not an empty dataset.",
            UserWarning,
            stacklevel=2,
        )
        return None

    return None


def _decode_dataset_bytes(raw: Any, encoding: Optional[str] = None) -> Optional[str]:
    """Decode pre-loaded dataset bytes, or refuse and say why.

    BGL5 AUDIT, 2026-09-27. This was ``raw.decode("utf-8", errors="replace")``,
    a silent data-corruption switch on the exact column a fairness analysis
    groups by. Measured on a latin-1 CSV (what Excel exports) whose group column
    holds the two distinct values 0xC4 and 0xC5::

        before: {'grp': ['\\ufffd', '\\ufffd', '\\ufffd', '\\ufffd'], 'y': [1, 0, 1, 0]}
                1 distinct group where the file holds 2, warnings []
        after:  None, with a UserWarning naming byte 5 and the encoding switch

    Two distinct protected groups merged into one, which is a disparity of
    exactly 0.0 between groups that were never compared. And on a non-CSV blob
    (gzip bytes, the stand-in for a still-encrypted or parquet payload the
    consumer was supposed to decode) the replacement characters produced a 1x1
    DataFrame whose column name was mojibake, so the handlers' documented
    ``data is None`` refusal never fired and the mediation ran on one garbage
    cell. Both are refusals now.

    ``encoding`` is the deliberate escape hatch: a caller that KNOWS the export
    is cp1252 says so, and the bytes are decoded strictly in that encoding, so a
    real non-UTF-8 dataset is still loadable without silently mangling an
    unknown one.
    """
    # EVERY BUFFER TYPE, not just ``bytes``. BGL6 AUDIT, 2026-09-29. This opened
    # with ``if not isinstance(raw, bytes): return str(raw)``, and a ``bytearray``
    # and a ``memoryview`` are NOT ``bytes``, so both skipped the whole strict
    # decode below and were turned into their own repr. The consumer decrypts the
    # stored file and attaches the result, which is exactly where a bytearray or a
    # memoryview comes from. Measured on the same latin-1 CSV the docstring pins:
    #
    #   bytes(latin)      -> None + 1 warning                        (correct)
    #   bytearray(latin)  -> a (0, 4) frame whose columns are
    #                        ["bytearray(b'grp", 'y\\n\\xc4', '1\\n\\xc5', "0\\n')"],
    #                        0 warnings
    #   memoryview(latin) -> a (0, 1) frame named '<memory at 0x...>', 0 warnings
    #   gzip bytes        -> None + 1 warning                        (correct)
    #   bytearray(gzip)   -> a (0, 1) frame named with the repr, 0 warnings
    #   a VALID utf-8 CSV as bytearray -> a (0, 4) frame of repr fragments, so
    #                        every row was silently lost
    #
    # so the handlers' documented ``data is None`` refusal never fired and a causal
    # op ran on an empty frame of repr fragments. ``_parse_csv_text`` could not
    # catch it either: shape[1] != 0 and the repr's \\x escapes are printable text,
    # not control characters. It now refuses a frame with no ROWS as well, which is
    # the backstop for any other type that reaches the ``str(raw)`` fallback below.
    if isinstance(raw, (bytearray, memoryview)):
        raw = bytes(raw)
    if not isinstance(raw, bytes):
        return str(raw)
    for candidate in ([encoding] if encoding else []) + ["utf-8"]:
        try:
            text = raw.decode(candidate)
        except (UnicodeDecodeError, LookupError) as exc:
            warnings.warn(
                f"load_dataset_from_payload: the {len(raw)} pre-loaded byte(s) are not "
                f"valid {candidate} ({exc}), so NOTHING was loaded. They are not decoded "
                f"with errors='replace': that turns every undecodable byte into the same "
                f"U+FFFD, which merges distinct group labels into one and reports a "
                f"disparity of 0.0 between groups that were never compared. Re-export the "
                f"file as UTF-8, or pass payload['dataset_encoding'] naming the encoding "
                f"it really is.",
                UserWarning,
                stacklevel=3,
            )
            return None
        if "\x00" in text:
            # Decoded, but not text: a NUL byte does not occur in a CSV export.
            # This is the parquet / still-encrypted / compressed case that used to
            # arrive as a 1x1 frame of mojibake.
            warnings.warn(
                f"load_dataset_from_payload: the {len(raw)} pre-loaded byte(s) decoded "
                f"but contain NUL bytes, so this is binary data and not a CSV export "
                f"(a still-encrypted, compressed or parquet payload reaches this function "
                f"when the consumer's decode step did not run). NOTHING was loaded; a "
                f"single-cell frame of mojibake is not a dataset.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return text
    return None


def _parse_csv_text(text: str, where: str) -> Optional[pd.DataFrame]:
    """Parse CSV text, or refuse and say why.

    BGL5, 2026-09-27. A parse failure used to escape as an exception into each
    handler's blanket ``except``, and a blob that PARSED into a 1x1 frame of
    mojibake was accepted outright. A frame with no columns, no rows, or whose
    column names carry control characters, is not a CSV, and the honest answer is
    the same None the handlers already refuse on.
    """
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
        warnings.warn(
            f"load_dataset_from_payload: {where} could not be parsed as CSV ({exc}), so "
            f"NOTHING was loaded.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if frame.shape[1] == 0:
        warnings.warn(
            f"load_dataset_from_payload: {where} parsed to a frame with no columns, so "
            f"NOTHING was loaded.",
            UserWarning,
            stacklevel=3,
        )
        return None
    unprintable = [c for c in frame.columns if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(c))]
    if unprintable:
        warnings.warn(
            f"load_dataset_from_payload: {where} parsed to column name(s) carrying control "
            f"characters ({unprintable!r}), which a CSV header does not, so this is binary "
            f"data rather than a dataset. NOTHING was loaded.",
            UserWarning,
            stacklevel=3,
        )
        return None
    # NO ROWS is the same could-not-load as no columns, and it is the shape every
    # accidental repr arrives in. BGL6 AUDIT, 2026-09-29: a bytearray or memoryview
    # payload was stringified into its own repr, which pandas read as a single
    # HEADER line, so ``bytearray(b'grp,y\na,1\nb,0\n')`` parsed to a (0, 4) frame
    # of repr fragments and was returned as a dataset with zero warnings. The
    # decode path above is fixed, and this is the backstop: a frame carrying a
    # header and not one observation cannot support a causal estimate, and an
    # op running on it reports effects computed from nothing.
    if len(frame) == 0:
        warnings.warn(
            f"load_dataset_from_payload: {where} parsed to a frame with {frame.shape[1]} "
            f"column(s) and NO ROWS ({list(frame.columns)[:4]!r}), so NOTHING was loaded. A "
            f"header with no observations cannot support a causal estimate, and it is also "
            f"the shape a non-CSV object takes when it is stringified into its own repr "
            f"instead of being decoded.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return frame


def _gml_from_payload(payload: Dict[str, Any]) -> str:
    serialized = payload.get("serialized") or {}
    gml = serialized.get("gml")
    if not isinstance(gml, str) or not gml:
        raise ValueError("payload.serialized.gml is required for causal ops.")
    return gml


def _json_safe(value: Any, where: str, unmeasured: List[str]) -> Any:
    """Return ``value`` with every non-finite number replaced by ``None``.

    Each replacement is RECORDED in ``unmeasured`` under its dotted path, so the
    envelope can name the fields rather than quietly nulling them.

    BGL-G02 (2026-09-30). ``main()`` serialises with ``json.dumps(envelope,
    default=str)``, which writes the BARE TOKEN ``NaN`` for a non-finite float.
    That is not JSON: the Node consumer's ``JSON.parse`` throws on it and the
    whole result is lost, as a transport error rather than as a data problem.
    Reached with an ordinary CSV holding ``inf`` in one outcome cell, which is
    what a divide-by-zero export looks like::

        handle_mediate(...) -> {"success": true, ... "total_effect": NaN ...}
        json.dumps(envelope, allow_nan=False) -> ValueError: Out of range float

    ``total_effect`` was NaN in the very field a reader takes for the measured
    total effect, while the decomposition's own note already said the estimator
    returned no finite value. Sanitised HERE, in the one funnel all five
    handlers return through, rather than in five places.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {
            k: _json_safe(v, f"{where}.{k}" if where else str(k), unmeasured)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, f"{where}[{i}]", unmeasured) for i, v in enumerate(value)]
    if isinstance(value, numbers.Real):
        if not is_measured(value):
            unmeasured.append(where or "<root>")
            return None
        if not isinstance(value, (int, float)):
            # np.float32 / np.int64 / Decimal: json.dumps cannot write them and
            # ``default=str`` would put the digits in a STRING where the reader
            # expects a number.
            return float(value)
    return value


def _ok(data: Any) -> Dict[str, Any]:
    unmeasured: List[str] = []
    envelope: Dict[str, Any] = {"success": True, "data": _json_safe(data, "", unmeasured)}
    if unmeasured:
        envelope["unmeasured_fields"] = unmeasured
        envelope["unmeasured_note"] = (
            "The field(s) listed in unmeasured_fields held a non-finite number (NaN or an "
            "infinity) and are reported as null. They are NOT measured values; read the "
            "notes / warnings beside them for why. They are nulled here because the bare "
            "NaN token that would otherwise be written is not valid JSON."
        )
    return envelope


def _err(message: str) -> Dict[str, Any]:
    return {"success": False, "error": message}


# Handlers


def handle_identify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """vfairness_causal_identify - graph-only, no dataset needed."""
    try:
        gml = _gml_from_payload(payload)
        serialized = payload["serialized"]
        treatments = serialized.get("treatments") or []
        outcomes = serialized.get("outcomes") or []
        result = identify_paths(gml, treatments, outcomes)
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(f"identify failed: {exc}")


def handle_mediate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """vfairness_causal_mediate - NDE/NIE decomposition."""
    try:
        gml = _gml_from_payload(payload)
        serialized = payload["serialized"]
        data = load_dataset_from_payload(payload)
        if data is None:
            return _err(
                "Mediation requires a dataset. Attach inline_csv, inline_json, or pre-load dataset_bytes on the consumer."
            )
        treatments = serialized.get("treatments") or []
        outcomes = serialized.get("outcomes") or []
        mediators = serialized.get("mediators") or []
        # BGL-G02 (2026-09-30). An empty list here produced
        # {"success": true, "data": {"decompositions": [], "warnings": []}}: a
        # successful mediation analysis reporting nothing, from a request that
        # named nothing to analyse. Nothing ran, and neither the decomposition
        # list nor the warning list said so. The identify handler's op already
        # refuses this ("No treatments or no outcomes provided"); the mediation
        # triple loop simply iterated over nothing.
        empty = [
            name
            for name, value in (
                ("treatments", treatments),
                ("outcomes", outcomes),
                ("mediators", mediators),
            )
            if not value
        ]
        if empty:
            return _err(
                "mediate requires a non-empty serialized."
                + ", serialized.".join(empty)
                + ": with none named there is no (treatment, mediator, outcome) triple to "
                "decompose, and an empty decomposition list would read as a mediation "
                "analysis that found nothing."
            )
        result = decompose_mediation(gml, data, treatments, outcomes, mediators)
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(f"mediate failed: {exc}")


def handle_refute(payload: Dict[str, Any]) -> Dict[str, Any]:
    """vfairness_causal_refute - placebo / random common cause / subset / dummy outcome."""
    try:
        gml = _gml_from_payload(payload)
        options = payload.get("options") or {}
        treatment = options.get("treatment")
        outcome = options.get("outcome")
        if not treatment or not outcome:
            return _err("refute requires options.treatment and options.outcome.")
        data = load_dataset_from_payload(payload)
        if data is None:
            return _err("Refutation requires a dataset.")
        result = run_refutation_suite(gml, data, treatment, outcome)
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(f"refute failed: {exc}")


def handle_counterfactual(payload: Dict[str, Any]) -> Dict[str, Any]:
    """vfairness_causal_counterfactual - GCM individual-level counterfactual."""
    try:
        gml = _gml_from_payload(payload)
        options = payload.get("options") or {}
        treatment = options.get("treatment")
        outcome = options.get("outcome")
        factual = options.get("factual")
        intervention_value = options.get("intervention_value")
        individual_id = options.get("individual_id")
        if not treatment or not outcome or not isinstance(factual, dict):
            return _err(
                "counterfactual requires options.treatment, options.outcome, options.factual."
            )
        data = load_dataset_from_payload(payload)
        if data is None:
            return _err("Counterfactual requires a dataset to fit the structural causal model.")
        result = compute_counterfactual(
            gml, data, treatment, outcome, factual, intervention_value, individual_id
        )
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(f"counterfactual failed: {exc}")


def handle_attribute(payload: Dict[str, Any]) -> Dict[str, Any]:
    """vfairness_causal_attribute - drift / distribution-change attribution."""
    try:
        gml = _gml_from_payload(payload)
        options = payload.get("options") or {}
        outcome = options.get("outcome")
        if not outcome:
            return _err("attribute requires options.outcome.")
        # Two-sample design: baseline + current. Consumer is expected to put
        # them under dataset_ref.inline_json_baseline / inline_json_current,
        # OR pre-load both as DataFrames in payload['baseline_df'] /
        # payload['current_df'].
        baseline = payload.get("baseline_df")
        current = payload.get("current_df")
        if baseline is None or current is None:
            ref = payload.get("dataset_ref") or {}
            b = ref.get("inline_json_baseline")
            c = ref.get("inline_json_current")
            if b and c:
                baseline = pd.DataFrame(b)
                current = pd.DataFrame(c)
        if baseline is None or current is None:
            return _err("attribute requires baseline + current frames.")
        result = attribute_distribution_change(gml, baseline, current, outcome)
        return _ok(result.to_dict())
    except Exception as exc:
        return _err(f"attribute failed: {exc}")


# CLI dispatch
# Lets the Node consumer call us with `python -m vfairness.operations.causal.task_handlers <task_type>`
# and pipe the JSON payload over stdin. Returns the JSON envelope on stdout.

_HANDLERS = {
    "vfairness_causal_identify": handle_identify,
    "vfairness_causal_mediate": handle_mediate,
    "vfairness_causal_refute": handle_refute,
    "vfairness_causal_counterfactual": handle_counterfactual,
    "vfairness_causal_attribute": handle_attribute,
}


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(json.dumps(_err("Usage: python -m ... <task_type>")))
        sys.exit(2)

    task_type = sys.argv[1]
    handler = _HANDLERS.get(task_type)
    if handler is None:
        print(json.dumps(_err(f"Unknown task_type: {task_type}")))
        sys.exit(2)

    payload_text = sys.stdin.read() or "{}"
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        print(json.dumps(_err(f"Invalid JSON payload: {exc}")))
        sys.exit(2)

    envelope = handler(payload)
    # allow_nan=False, deliberately: the default writes the bare tokens NaN /
    # Infinity, which JSON.parse on the consumer side rejects, so a result the
    # handler believed it had delivered never arrives. _ok() sanitises and NAMES
    # any non-finite field, so reaching this ValueError means something got past
    # that funnel, and saying so is better than emitting invalid JSON.
    try:
        print(json.dumps(envelope, default=str, allow_nan=False))
    except (ValueError, TypeError) as exc:
        print(
            json.dumps(
                _err(
                    f"{task_type} produced a result that could not be serialised as valid "
                    f"JSON ({exc}). Nothing is reported rather than a payload the consumer "
                    f"cannot parse."
                )
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
