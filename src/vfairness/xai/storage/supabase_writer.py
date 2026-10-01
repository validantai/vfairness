"""
SupabaseWriter -- inserts vfairness.xai output rows into the canonical
assessment-layer tables.

Reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from the environment.
The service-role key is the only way the worker can bypass owner-scoped
RLS to insert rows on behalf of the customer. Owner is always passed
explicitly through to the row (every assessment table has an ``owner``
text column == auth0_sub of the customer who owns the run).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict
from typing import Any

from ..schemas import Explanation, FairnessDecomposition, XaiAssessment


class SupabaseWriterError(RuntimeError):
    """Surfaced when the Supabase insert / update fails."""


class SupabaseWriter:
    """Thin wrapper around supabase-py for the XAI assessment-layer tables."""

    def __init__(self, url: str | None = None, service_role_key: str | None = None) -> None:
        try:
            from supabase import create_client
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vfairness.xai.storage.SupabaseWriter requires `supabase`: "
                "`pip install vfairness[worker]`."
            ) from exc

        self._url = url or os.environ.get("SUPABASE_URL")
        self._key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not self._url or not self._key:
            raise SupabaseWriterError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in env or constructor."
            )
        self._client = create_client(self._url, self._key)

    # single-row inserts
    def write_audit_artifact(
        self,
        *,
        owner: str,
        subject_id: str,
        data_hash: str,
        model_hash: str,
        params: dict[str, Any],
        library_versions: dict[str, str],
        seed: int | None,
        artifact_uri: str,
    ) -> str:
        """Insert one row into xai_audit_artifacts; returns the new id."""
        row = {
            "owner": owner,
            "subject_id": subject_id,
            "data_hash": data_hash,
            "model_hash": model_hash,
            "params": params,
            "library_versions": library_versions,
            "seed": seed,
            "artifact_uri": artifact_uri,
        }
        return self._insert_one("xai_audit_artifacts", row)

    def write_explanations(
        self,
        owner: str,
        explanations: list[Explanation],
    ) -> list[str]:
        rows = [e.to_db_row(owner) for e in explanations]
        return self._insert_many("xai_explanations", rows)

    def write_fairness_decomposition(
        self,
        owner: str,
        decomposition: FairnessDecomposition,
    ) -> str:
        # ``to_db_row`` re-asserts the 1e-6 identity; broken rows never reach the DB.
        return self._insert_one("xai_fairness_decompositions", decomposition.to_db_row(owner))

    def write_assessment(self, owner: str, assessment: XaiAssessment) -> str:
        # Writes the XaiAssessment summary; the individual Explanation rows
        # are inserted separately via write_explanations.
        row = {
            "owner": owner,
            "subject_id": assessment.subject_id,
            "explainer": assessment.explainer,
            "scope": assessment.scope,
            "audit_artifact_id": assessment.audit_artifact_id,
            "view_provenance": assessment.view_provenance,
            "diagnostics": asdict(assessment.diagnostics),
            "sp_shap_representatives": assessment.sp_shap_representatives,
        }
        return self._insert_one("xai_assessments", row)

    # job lifecycle
    def update_job_progress(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        message: str | None = None,
    ) -> int | None:
        """Record job progress, and say so when no row was actually updated.

        Returns how many rows the update matched, or ``None`` when the response
        carried no representation and the question cannot be answered.

        BGL3 xai-1, 2026-09-27. The response was discarded, and PostgREST
        returns an EMPTY data list when the ``id`` filter matches nothing.
        Measured with a stubbed client, the same call against a row that exists
        and against an id that does not:

            existing id  -> returned None, resp.data == [{"id": "job-1", ...}]
            unknown id   -> returned None, resp.data == []

        identical from the caller's side, with no warning. So a worker writing
        "status failed, progress 100" into a job row that is not there recorded a
        clean handover while the row kept whatever it last said, and the only
        evidence either way was thrown away one expression earlier. This matters
        here in particular: ``WorkerLoop._dispatch`` passes the pgmq ``msg_id``
        when the payload has no ``job_id``, and a queue message id is not an
        ``xai_jobs`` primary key, so the zero-row update is a live path and not a
        hypothetical one.

        ``_insert_one`` below already refuses a zero-row write. Three states
        rather than two: matched (returns the count), provably not matched
        (raises, because an unrecorded status is not progress), and unconfirmable
        (warns and returns ``None``, which is not success).
        """
        resp = (
            self._client.table("xai_jobs")
            .update(
                {
                    "status": status,
                    "progress": progress,
                    "message": message,
                }
            )
            .eq("id", job_id)
            .execute()
        )
        data = getattr(resp, "data", None)
        if data is None:
            warnings.warn(
                "SupabaseWriter.update_job_progress: the update of job "
                f"{job_id} returned no representation, so whether any row was written COULD "
                "NOT BE CONFIRMED. Returning None, which is not a successful update.",
                UserWarning,
                stacklevel=2,
            )
            return None
        if not data:
            raise SupabaseWriterError(
                f"update of xai_jobs matched no row for id {job_id} (0 rows written), so "
                f"status={status!r} progress={progress} was NOT recorded anywhere."
            )
        return len(data)

    # internals
    def _row_id(self, table: str, row: Any, position: int, n_rows: int) -> str:
        """The id PostgREST returned for one written row, or a named refusal.

        ``row["id"]`` raised a bare ``KeyError: 'id'`` for a response row that
        carries no primary key, from a line that says nothing about which table
        or which position, so the caller could not tell it from a bug in its own
        code. The row itself is NOT put in the message: it can echo the
        attempted data, which is the reason the two refusals above withhold the
        response repr as well.
        """
        if not isinstance(row, dict) or "id" not in row:
            raise SupabaseWriterError(
                f"insert into {table} returned a row at position {position} of {n_rows} with "
                f"no 'id' key, so the written row cannot be identified. The insert may or may "
                f"not have persisted; treat this as unconfirmed, not as a success."
            )
        return str(row["id"])

    def _insert_one(self, table: str, row: dict[str, Any]) -> str:
        resp = self._client.table(table).insert(row).execute()
        if not resp.data:
            # Do not embed the response repr: it can echo the attempted row data.
            raise SupabaseWriterError(f"insert into {table} returned no data (0 rows written)")
        return self._row_id(table, resp.data[0], 0, 1)

    def _insert_many(self, table: str, rows: list[dict[str, Any]]) -> list[str]:
        """Insert *rows* and return one id per row, or refuse.

        A PARTIAL WRITE USED TO PASS WHERE A TOTAL ONE WAS REFUSED. ``if not
        resp.data`` catches only the all-or-nothing case, so the counts were
        checked at zero and nowhere else. Measured 2026-09-30 against a stubbed
        client, ten Explanation rows in:

            10 returned  -> 10 ids, no warning      (correct)
             0 returned  -> SupabaseWriterError naming "0 of 10"
             3 returned  ->  3 ids, NO WARNING

        A worker calling ``write_explanations`` got a shorter list back and
        nothing told it that seven explanations are not in the database. The
        list length is not a signal a caller can read either: it never knew how
        many rows the producer had, and the natural next step, storing the ids,
        succeeds with three of them. So the shortfall is refused with BOTH
        counts, exactly as the zero case already was.

        MORE rows than were sent is a different fact: nothing was lost, but the
        response does not describe the request, so it warns rather than raising.
        """
        if not rows:
            return []
        resp = self._client.table(table).insert(rows).execute()
        data = getattr(resp, "data", None)
        if not data:
            # Do not embed the response repr: it can echo the attempted row data.
            raise SupabaseWriterError(
                f"bulk insert into {table} returned no data (0 of {len(rows)} rows written)"
            )
        if len(data) < len(rows):
            raise SupabaseWriterError(
                f"bulk insert into {table} wrote only {len(data)} of {len(rows)} row(s). The "
                f"remaining {len(rows) - len(data)} are NOT in the database, and returning the "
                f"{len(data)} id(s) would read as a complete write."
            )
        if len(data) > len(rows):
            warnings.warn(
                f"SupabaseWriter: bulk insert into {table} returned {len(data)} row(s) for "
                f"{len(rows)} sent. Nothing was lost, but the response does not describe the "
                f"request, so the id list may not correspond to the rows supplied.",
                UserWarning,
                stacklevel=3,
            )
        return [self._row_id(table, r, i, len(data)) for i, r in enumerate(data)]
