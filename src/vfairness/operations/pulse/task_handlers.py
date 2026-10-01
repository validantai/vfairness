"""Pulse task handler -- registered for ``vfairness_pulse_run``.

THIN SHIM ONLY. This handler resolves the dataset artifact and calls
:func:`vfairness.operations.pulse.orchestrator.run_pulse`, the single
source of truth for ALL Pulse analysis. It must never compute metrics,
bin attributes, or shape the verdict itself: parallel implementations
drift and have produced fabricated output before (the legacy
``pipeline.run_pulse_pipeline`` invented equalized-odds/parity numbers
when labels were absent; it is quarantined and must not be re-wired).

The browser dispatches a single task type with the captured inputs and a
signed URL pointing at the ZK-encrypted artifact in the ``pulse-artifacts``
bucket. The consumer:

    1. Downloads the bytes from ``payload['artifact']['signed_url']``.
    2. If the envelope is the v2 ZK shape, decrypts with the org key.
    3. Parses CSV / JSONL / Parquet.
    4. Invokes :func:`vfairness.operations.pulse.orchestrator.run_pulse`
       with the FULL ``inputs`` dict passed through verbatim (source_kind,
       llm_config, min_group_size, outcome_polarity, artifact_hash,
       reference_distribution, ... -- the orchestrator owns the contract;
       the shim must not filter keys).
    5. Returns ``{"success": True, "data": result}`` where ``data`` matches
       the canonical PulseResult shape the React UI renders.

A consumer registration table lives in ``CONSUMER_REGISTRATION.md``.
"""

from __future__ import annotations

import base64
import io
import json
import ssl
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

import pandas as pd

from vfairness.result import TaskResult

# G-10: bind the trace-ingestion front door. It flattens raw
# OTel/Langfuse span exports into the per-episode trace table and then
# delegates every frame to orchestrator.run_pulse, which remains the
# single source of truth for ALL Pulse analysis.
from .traces import ingest_and_run_pulse as run_pulse

# Verified-CA SSL context. macOS framework Python ships without a usable CA
# bundle, so a bare urlopen() to the Supabase signed URL fails with
# CERTIFICATE_VERIFY_FAILED. Prefer certifi; fall back to the system default.
try:
    import certifi as _certifi

    _SSL_CTX: "ssl.SSLContext" = ssl.create_default_context(cafile=_certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()


_TASK_TYPE = "vfairness_pulse_run"


# Artifact loaders


def _download_bytes(url: str, timeout: float = 60.0, max_bytes: int = 1024 * 1024 * 1024) -> bytes:
    # Defence in depth. The signed artifact URL is first-party, but pin the
    # scheme to https so a malformed/hostile payload cannot downgrade this into a
    # file:// local read or an http:// fetch, and cap the read so an oversized
    # artifact cannot exhaust worker memory. Raise max_bytes for larger datasets.
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme != "https":
        raise ValueError(f"artifact URL must use https, got scheme {scheme!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "validant-pulse/1"})
    # B310: scheme is pinned to https above and the CA context is pinned to
    # certifi; the URL is a first-party signed artifact URL, not attacker input.
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:  # nosec B310
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(
            f"artifact exceeds the {max_bytes}-byte download cap; raise max_bytes if this is expected"
        )
    return data


def _decrypt_envelope(raw: bytes, org_key_b64: Optional[str]) -> bytes:
    """If the file looks like the ZK v2 envelope, decrypt with the org key.

    The browser uploads either:
      - Plain bytes (when no org key is set up), or
      - JSON envelope ``{v: 2, certificate, orgId, uploadedAt, payload}`` where
        ``payload`` is the AES-GCM v2 EncryptedData blob.
    """
    if not raw:
        return raw
    # Quick sniff: JSON envelopes always start with ``{``.
    head = raw[:1]
    if head != b"{":
        return raw
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw
    if envelope.get("v") != 2 or "payload" not in envelope:
        return raw
    if not org_key_b64:
        raise RuntimeError(
            "Artifact is ZK-encrypted but no org_key_b64 was supplied to the "
            "consumer. Pulse cannot decrypt the dataset."
        )
    plaintext_b64 = _decrypt_with_key(envelope["payload"], org_key_b64)
    return base64.b64decode(plaintext_b64)


def _decrypt_with_key(encrypted: Dict[str, Any], org_key_b64: str) -> str:
    """AES-GCM v2 decryption matching the browser ``encryptWithKey`` output.

    The browser stores:
        {"v": 2, "iv": "<base64>", "ct": "<base64>", "key_id": "<hex>"}

    Older shape (v1) carries salt + iteration count and uses PBKDF2 -- the
    Pulse upload path is v2 only, so the v1 branch is left as a future
    extension.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Decrypting a ZK-encrypted Pulse artifact requires the `cryptography` "
            "package. Install it (e.g. `pip install vfairness[pulse]`) to run the "
            "encrypted-artifact Pulse flow."
        ) from exc

    key = base64.b64decode(org_key_b64)
    iv = base64.b64decode(encrypted["iv"])
    # Browser writes `ciphertext` (the `encryptWithKey` shape documented above).
    # Accept `ct` as well for forward compatibility with any future shape change.
    ct_b64 = encrypted.get("ciphertext") or encrypted.get("ct")
    if not ct_b64:
        raise RuntimeError("ZK envelope missing 'ciphertext' field.")
    ct = base64.b64decode(ct_b64)
    aes = AESGCM(key)
    plaintext = aes.decrypt(iv, ct, None)
    return plaintext.decode("utf-8")


def _parse_dataset(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    text = None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(io.BytesIO(raw))
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if name.endswith(".json"):
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        data = json.loads(text)
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError("JSON artifacts must be an array of row objects.")
    # Default: CSV
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


# Handler


def handle_pulse_run(payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
    """Entry point for the ``vfairness_pulse_run`` task type.

    Thin shim: resolve the artifact, forward the FULL inputs dict to
    ``run_pulse``. Never filter inputs keys here -- the orchestrator owns
    the contract and new keys (source_kind, llm_config, outcome_polarity,
    min_group_size, artifact_hash, ...) must flow through untouched.

    ``progress`` is the optional per-stage callback
    ``progress(stage, label, stage_index, total_stages, pct)`` a consumer
    may pass to surface live telemetry; it is forwarded verbatim.
    """
    try:
        inputs = dict(payload.get("inputs") or {})
        artifact = payload.get("artifact") or {}
        signed_url = artifact.get("signed_url")
        filename = artifact.get("filename") or "artifact.csv"
        org_key_b64 = payload.get("org_key_b64") or artifact.get("org_key_b64")

        # 1. Resolve dataset. Endpoint-only runs (live LLM probe) carry no
        #    dataset at all: the orchestrator's B1 branch probes the
        #    endpoint directly, so an empty frame is passed through.
        _src = str(inputs.get("source_kind") or inputs.get("sourceKind") or "").lower()
        endpoint_only = bool(
            inputs.get("llm_config")
            or inputs.get("endpoint")
            or _src in ("endpoint", "llm", "model_endpoint")
        )
        df: Optional[pd.DataFrame] = None
        if signed_url:
            raw = _download_bytes(signed_url)
            decrypted = _decrypt_envelope(raw, org_key_b64)
            df = _parse_dataset(decrypted, filename)
        elif artifact.get("inline_csv"):
            df = pd.read_csv(io.StringIO(artifact["inline_csv"]))
        elif artifact.get("inline_json"):
            df = pd.DataFrame(artifact["inline_json"])
        elif endpoint_only:
            df = pd.DataFrame()
        else:
            return TaskResult.fail(
                _TASK_TYPE,
                (
                    "No artifact provided. Pulse needs either a signed URL or an "
                    "inline CSV / JSON payload."
                ),
            ).to_dict()

        if (df is None or df.empty) and not endpoint_only:
            return TaskResult.fail(_TASK_TYPE, "Artifact decoded but contained no rows.").to_dict()

        # 2. Run the canonical orchestrator (full inputs pass-through).
        #    run_pulse owns the {"success": ..., "data": ...} contract and
        #    remains the single source of truth. We adopt its envelope via
        #    TaskResult.from_envelope so the schema_version/task_type keys are
        #    attached WITHOUT re-nesting its `data` or dropping any key.
        if not inputs.get("artifact_label") and not inputs.get("artifactLabel"):
            inputs["artifact_label"] = filename
        envelope = run_pulse(df, inputs, progress=progress)
        return TaskResult.from_envelope(_TASK_TYPE, envelope).to_dict()

    except Exception as exc:  # pragma: no cover -- consumer surfaces this verbatim
        return TaskResult.fail(_TASK_TYPE, f"Pulse run failed: {exc}").to_dict()


# CLI entry (subprocess registration option)


def main() -> int:
    """``python -m vfairness.operations.pulse.task_handlers vfairness_pulse_run``.

    Reads a JSON payload from stdin, writes the envelope to stdout. Exit code
    is 0 on a successful envelope (which may still wrap ``success: false``).
    """
    task_type = sys.argv[1] if len(sys.argv) > 1 else "vfairness_pulse_run"
    payload_text = sys.stdin.read()
    try:
        payload = json.loads(payload_text) if payload_text.strip() else {}
    except json.JSONDecodeError as exc:
        json.dump(
            TaskResult.fail(_TASK_TYPE, f"invalid payload JSON: {exc}").to_dict(),
            sys.stdout,
        )
        return 1

    if task_type != _TASK_TYPE:
        json.dump(
            TaskResult.fail(task_type, f"unknown task type: {task_type}").to_dict(),
            sys.stdout,
        )
        return 1

    envelope = handle_pulse_run(payload)
    json.dump(envelope, sys.stdout, default=str)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
