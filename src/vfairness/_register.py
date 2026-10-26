"""Model Register support: fingerprints and evidence bundles (pre-release).

PRE-RELEASE, OUTSIDE THE FROZEN PUBLIC SURFACE. This module is private on
purpose: it has not been through the library's grading yet, so it is not part
of the public API that the capability ledger and the beta gate count. It
becomes the public ``vfairness.register`` once its units are graded. Until
then its names may change.

A model version is identified by one SHA-256 digest of its bytes, computed
where the model lives (your machine, your pipeline). Nothing here uploads the
model or the data: the fingerprint is the only thing that needs to leave.

Rules, shared with the Validant platform so both sides compute the same digest:

* A single file (a dataset, a model file): SHA-256 of the file's bytes.
* A folder (an MLflow model directory, an unpacked archive): the *manifest
  digest*. For every regular file below the folder, take its path relative to
  the folder with ``/`` separators, then a NUL, then the lowercase hex SHA-256
  of its bytes, then a newline. Sort these lines by the UTF-8 bytes of the
  path and take the SHA-256 of all of them together.

The evidence bundle (format ``validant.register.evidence.v1``) is what a
pipeline hands to the Validant Model Register: One JSON document per assessed model version (format
``validant.register.evidence.v1``). You import it in the Validant platform,
where your browser encrypts the ``mlflow`` and ``results`` parts with your
organisation's key before anything is sent. Only the fingerprint, the dataset
fingerprint, the system type and the pathway are stored in clear, so Validant
can sign a receipt for exactly these bytes without seeing the model, the data,
the model's name or the results.

Example::

    from vfairness._register import model_fingerprint, dataset_fingerprint, build_evidence_bundle, write_evidence_bundle

    fp = model_fingerprint("models:/credit-scoring@candidate")
    bundle = build_evidence_bundle(fp, dataset_sha256=dataset_fingerprint("holdout.csv").digest,
                                   mlflow={"model_uri": "models:/credit-scoring@candidate"})
    write_evidence_bundle(bundle, "credit-scoring.evidence.json")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Fingerprint:
    """One identity: always SHA-256 here, with the number of files it covers."""

    digest: str
    files: int
    method: str = "sha256"

    def as_dict(self) -> dict:
        return {"method": self.method, "digest": self.digest, "files": self.files}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _normalise(path: str) -> str:
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"Manifest path is not inside the folder: {path!r}")
    return "/".join(parts)


def manifest_digest(files: Iterable[tuple[str, bytes | str]]) -> str:
    """The manifest digest of ``(relative path, bytes or file SHA-256)`` pairs.

    A second element of exactly 64 lowercase hex characters given as ``str``
    is taken as the file's SHA-256 already computed; ``bytes`` are hashed.
    """
    entries: list[tuple[bytes, str]] = []
    for path, content in files:
        if isinstance(content, str):
            if len(content) != 64 or any(c not in "0123456789abcdef" for c in content):
                raise ValueError(f"Not a lowercase SHA-256 hex digest for {path!r}")
            sha = content
        else:
            sha = hashlib.sha256(content).hexdigest()
        entries.append((_normalise(path).encode("utf-8"), sha))
    if not entries:
        raise ValueError("A folder with no files has no fingerprint.")
    entries.sort(key=lambda e: e[0])
    for (a, _), (b, _) in zip(entries, entries[1:]):
        if a == b:
            raise ValueError(f"Manifest lists {a.decode('utf-8')} twice.")
    text = b"".join(p + b"\x00" + s.encode("ascii") + b"\n" for p, s in entries)
    return hashlib.sha256(text).hexdigest()


def file_fingerprint(path: str | os.PathLike) -> Fingerprint:
    """SHA-256 of one file's bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return Fingerprint(digest=_sha256_file(p), files=1)


def dataset_fingerprint(data: str | os.PathLike | bytes) -> Fingerprint:
    """SHA-256 of a dataset: a path to the file, or its bytes as read."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return Fingerprint(digest=hashlib.sha256(bytes(data)).hexdigest(), files=1)
    return file_fingerprint(data)


def folder_fingerprint(path: str | os.PathLike) -> Fingerprint:
    """The manifest digest of every regular file below a folder.

    Symbolic links are refused rather than followed, so the digest always
    describes bytes that are inside the folder.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    pairs: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        for name in dirnames:
            if (here / name).is_symlink():
                raise ValueError(f"Symbolic link in the model folder: {here / name}")
        for name in filenames:
            f = here / name
            if f.is_symlink():
                raise ValueError(f"Symbolic link in the model folder: {f}")
            if f.is_file():
                pairs.append((f.relative_to(root).as_posix(), _sha256_file(f)))
    return Fingerprint(digest=manifest_digest(pairs), files=len(pairs))


def model_fingerprint(model: str | os.PathLike) -> Fingerprint:
    """The fingerprint of a model: a local folder or file, or an MLflow model URI.

    For an MLflow URI (``models:/<name>/<version>``, ``models:/<name>@<alias>``,
    ``runs:/<run>/<path>``, or any artifact URI your MLflow can read) the model
    is downloaded with your own MLflow client into a temporary folder, hashed
    there, and the folder is removed. This is the only function here that talks
    to a server, and only to the MLflow you configured.
    """
    text = os.fspath(model)
    local = Path(text)
    if local.is_dir():
        return folder_fingerprint(local)
    if local.is_file():
        return file_fingerprint(local)
    if "://" not in text and not text.startswith(("models:/", "runs:/")):
        raise FileNotFoundError(text)
    try:
        import mlflow.artifacts as mlflow_artifacts
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Reading a model from MLflow needs the mlops extra: pip install 'vfairness[mlops]'"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="vfairness-fingerprint-") as tmp:
        downloaded = Path(mlflow_artifacts.download_artifacts(artifact_uri=text, dst_path=tmp))
        return (
            folder_fingerprint(downloaded) if downloaded.is_dir() else file_fingerprint(downloaded)
        )


__all__ = [
    "BUNDLE_FORMAT",
    "PATHWAYS",
    "SYSTEM_TYPES",
    "Fingerprint",
    "build_evidence_bundle",
    "dataset_fingerprint",
    "file_fingerprint",
    "folder_fingerprint",
    "manifest_digest",
    "model_fingerprint",
    "read_evidence_bundle",
    "write_evidence_bundle",
]

BUNDLE_FORMAT = "validant.register.evidence.v1"
SYSTEM_TYPES = ("predictive", "llm", "agent", "multi_agent")
#: Pathways whose identity is a SHA-256 of content; the others are keyed in the browser.
PATHWAYS = ("mlflow", "model_file", "data_only", "platform_trained")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MLFLOW_KEYS = ("model_uri", "model_id", "run_id", "version")


def _check_hex(value: str | None, field: str, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ValueError(f"{field} must be 64 lowercase hex characters")


def build_evidence_bundle(
    identity: Fingerprint,
    *,
    dataset_sha256: str | None = None,
    system_type: str = "predictive",
    pathway: str = "mlflow",
    mlflow: dict | None = None,
    results: dict | None = None,
    vfairness_version: str | None = None,
    created_at: datetime | None = None,
) -> dict:
    """Build and validate one evidence bundle.

    ``mlflow`` may hold ``model_uri``, ``model_id``, ``run_id`` and ``version``
    (other keys are refused, so nothing unexpected travels); ``results`` holds
    whatever assessment output you choose to keep with the version, for
    example a fairness report as ``report.to_dict()``.
    """
    if not isinstance(identity, Fingerprint) or identity.method != "sha256":
        raise ValueError(
            "identity must be a SHA-256 Fingerprint (model_fingerprint, folder_fingerprint)"
        )
    _check_hex(identity.digest, "identity.digest", True)
    _check_hex(dataset_sha256, "dataset_sha256", False)
    if system_type not in SYSTEM_TYPES:
        raise ValueError(f"system_type must be one of {SYSTEM_TYPES}")
    if pathway not in PATHWAYS:
        raise ValueError(
            f"pathway must be one of {PATHWAYS}; endpoints and links are identified in the browser"
        )
    mlflow = dict(mlflow or {})
    unknown = sorted(set(mlflow) - set(_MLFLOW_KEYS))
    if unknown:
        raise ValueError(f"unexpected mlflow keys: {unknown}")
    if vfairness_version is None:
        from vfairness import __version__ as vfairness_version
    stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bundle = {
        "format": BUNDLE_FORMAT,
        "system_type": system_type,
        "pathway": pathway,
        "identity": identity.as_dict(),
        "dataset_sha256": dataset_sha256,
        "vfairness_version": str(vfairness_version),
        "created_at": stamp.isoformat().replace("+00:00", "Z"),
        "mlflow": {k: (None if mlflow.get(k) is None else str(mlflow[k])) for k in _MLFLOW_KEYS},
        "results": results or {},
    }
    json.dumps(bundle, allow_nan=False)  # refuses NaN and anything not JSON before it is written
    return bundle


def write_evidence_bundle(bundle: dict, path: str | os.PathLike) -> Path:
    """Write the bundle as UTF-8 JSON; returns the path written."""
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError("not an evidence bundle")
    out = Path(path)
    out.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out


def read_evidence_bundle(path: str | os.PathLike) -> dict:
    """Read and re-validate a bundle written by ``write_evidence_bundle``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != BUNDLE_FORMAT:
        raise ValueError("not an evidence bundle")
    identity = data.get("identity") or {}
    fp = Fingerprint(
        digest=str(identity.get("digest", "")),
        files=int(identity.get("files", 0)),
        method=str(identity.get("method", "")),
    )
    rebuilt = build_evidence_bundle(
        fp,
        dataset_sha256=data.get("dataset_sha256"),
        system_type=data.get("system_type"),
        pathway=data.get("pathway"),
        mlflow={k: v for k, v in (data.get("mlflow") or {}).items() if v is not None},
        results=data.get("results"),
        vfairness_version=data.get("vfairness_version"),
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
    )
    return rebuilt
