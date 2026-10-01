"""
vfairness.xai.storage
=====================

Writers + artifact-bundle utilities for the canonical Supabase tables.

* :class:`SupabaseWriter` -- inserts into ``xai_explanations``,
  ``xai_fairness_decompositions``, ``xai_audit_artifacts``,
  ``xai_trust_postures``, and ``xai_jobs``. RLS-aware (passes the
  worker's service-role auth0_sub through to RPCs that demand it).
* :func:`build_audit_artifact_bundle` -- packs an Explanation + its
  source params, library_versions and seed into a deterministic JSON
  bundle that the regulator can replay byte-stably.
"""

from .audit_artifact import build_audit_artifact_bundle
from .supabase_writer import SupabaseWriter, SupabaseWriterError

__all__ = [
    "SupabaseWriter",
    "SupabaseWriterError",
    "build_audit_artifact_bundle",
]
