"""
AuditArtifact bundle builder.

Packs an Explanation (or list of Explanations) plus reproducibility
metadata into a JSON blob whose CONTENT is content-addressed, so a
regulator can replay it and get the same digest given the same model and
the same seed.

The digest covers content only. Wall-clock fields stay in the artifact,
where a regulator wants to see them, but are excluded from the hashed
payload: a content address that covers the clock can never match a
re-run, which makes the reproducibility it exists to attest unverifiable.
Until 2026-08-27 both ``generated_at`` and every ``Explanation.timestamp``
were inside the hash, so two calls one second apart on identical inputs
produced different sha256s and the sha, used as the natural key in
``xai_audit_artifacts.id``, could never collide on identical content.

Stored in the ``xai-audit-artifacts`` Supabase bucket, scoped by
auth0_sub folder prefix per the bucket policy the deploying application
declares in its own storage migration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ..schemas import Explanation, FairnessDecomposition

#: Paths carrying wall-clock time. Present in the returned bundle, absent from
#: the hashed payload. Declared inside the hashed content itself and echoed on
#: the bundle, so a verifier can reproduce the digest without reading this file.
DIGEST_EXCLUDED_PATHS = ("generated_at", "explanations[].timestamp")

SCHEMA_VERSION = "1.1"


def _content_explanation(explanation_dict: dict[str, Any]) -> dict[str, Any]:
    """One explanation dict with its wall-clock timestamp removed.

    The explanation's identity is already carried by ``model_hash`` /
    ``data_hash`` / ``method`` / ``params``, so dropping the clock loses no
    information the digest needs.
    """
    return {k: v for k, v in explanation_dict.items() if k != "timestamp"}


def build_audit_artifact_bundle(
    *,
    subject_id: str,
    explanations: list[Explanation],
    decomposition: FairnessDecomposition | None = None,
    params: dict[str, Any],
    library_versions: dict[str, str],
    seed: int | None,
) -> tuple[dict[str, Any], str]:
    """Build the audit artifact bundle.

    Returns ``(bundle, sha256)``. The sha256 is a CONTENT address: it is the
    natural key in ``xai_audit_artifacts.id`` (truncated to 16 chars for
    human-readable display), so identical content must produce an identical
    digest or the key cannot dedupe and a replay can never be checked.

    Determinism: keys are sorted, floats are not re-formatted, and every field
    in :data:`DIGEST_EXCLUDED_PATHS` is left out of the hashed payload. Two
    calls with the same explanations, params, library versions and seed return
    the same sha256; changing any attribution, param or seed changes it.

    ``generated_at`` and each explanation's ``timestamp`` remain on the returned
    bundle. They are evidence for the reader, not inputs to the address.
    """
    explanations_full = [asdict(e) for e in explanations]
    decomposition_dict = asdict(decomposition) if decomposition else None

    # What the digest covers. Anything not in here is, by construction, not
    # attested by the sha256.
    content = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": subject_id,
        "explanations": [_content_explanation(d) for d in explanations_full],
        "decomposition": decomposition_dict,
        "params": params,
        "library_versions": library_versions,
        "seed": seed,
        "digest_excludes": list(DIGEST_EXCLUDED_PATHS),
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=True).encode("utf-8")
    sha = hashlib.sha256(canonical).hexdigest()

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": subject_id,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "explanations": explanations_full,
        "decomposition": decomposition_dict,
        "params": params,
        "library_versions": library_versions,
        "seed": seed,
        "digest_excludes": list(DIGEST_EXCLUDED_PATHS),
        "content_sha256": sha,
    }
    return bundle, sha
