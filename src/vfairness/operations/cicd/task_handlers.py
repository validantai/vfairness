"""Task handler for ``vfairness_data_validation``.

Canonical data-quality entrypoint for any consumer (the Navigator's
"Run Quality Check", CI jobs, etc.). It runs the full five-dimension
``DataBiasValidator`` and returns the same plain-language report shape Pulse
renders, via the shared :func:`build_quality_report` -- one source of truth.

Payload contract
----------------
::

    {
      "csv_data": "<csv text>",                       # OR
      "inline_json": [ {..row..}, ... ],
      "protected_attributes": ["gender", "age"],
      "outcome_column": "approved"                     # optional
    }

Result envelope
---------------
Built via :class:`vfairness.result.TaskResult`. On success:
``{"schema_version", "task_type", "success": true, "data": {tone, headline,
rows, columns, checks}}``; on failure the ``data`` key is replaced by
``"error": "<message>"``. The ``success`` / ``data`` / ``error`` semantics are
unchanged from the historical shape; ``schema_version`` and ``task_type`` are
additive.

Exit code
---------
:func:`main` exits 0 only when the envelope says ``"success": true``. A refusal
(no dataset, no protected attribute present in the data) exits 1, like the bad
payload and unknown-task-type cases, because a CI step reads the exit code and
nothing else. See the measured before and after in :func:`main`.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any, Dict

import pandas as pd

from vfairness.result import TaskResult

from .quality_report import build_quality_report

_TASK_TYPE = "vfairness_data_validation"


def _load_df(payload: Dict[str, Any]) -> pd.DataFrame:
    if payload.get("csv_data"):
        return pd.read_csv(io.StringIO(payload["csv_data"]))
    if payload.get("inline_json"):
        return pd.DataFrame(payload["inline_json"])
    raise ValueError("No dataset provided: expected 'csv_data' or 'inline_json'.")


def handle_data_validation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the ``vfairness_data_validation`` task type."""
    try:
        df = _load_df(payload)
        if df.empty:
            return TaskResult.fail(_TASK_TYPE, "Dataset decoded but contained no rows.").to_dict()

        requested = list(
            payload.get("protected_attributes") or payload.get("protectedAttributes") or []
        )
        protected = [c for c in requested if c in df.columns]
        # BGL3 operations-4, 2026-09-27. The names dropped here were dropped
        # SILENTLY, and the report is what the caller reads. Measured on 400
        # rows with a real 2.6x gender gap, protected_attributes ['gender',
        # 'disability'] where 'disability' is not a column: the envelope was
        # byte-identical to the single-attribute request, with no row, note or
        # warning saying that half the requested scope was never assessed.
        # They are reported instead, at "warn", so the tone cannot stay green.
        unavailable = [c for c in requested if c not in df.columns]
        if not protected:
            return TaskResult.fail(
                _TASK_TYPE,
                (
                    "None of the protected attributes are present in the data, "
                    "so group fairness cannot be validated."
                ),
            ).to_dict()

        # CRITICAL (fail loud): this used to read
        #     if outcome and outcome not in df.columns: outcome = None
        # which turned a caller's typo into a silent narrowing of the check.
        # Measured 2026-09-27 on 400 rows where M is approved 80 percent of the
        # time and F 30 percent, a 2.6x raw outcome gap:
        #   outcome_column='approved' -> tone 'critical', "[critical] Raw
        #       outcome gap ... Outcome rates differ sharply between groups"
        #   outcome_column='aproved'  -> tone 'pass', "The data is fit for a
        #       reliable fairness read", three green tiles, 0 warnings
        # on the SAME rows. The gap was never compared, and the report said the
        # data was fine. The name now goes through to the canonical validator,
        # which answers `missing_outcome_column` for it, so the report carries a
        # row saying which checks could not run. One judgment, made in one
        # place, exactly as this module's header requires.
        outcome = payload.get("outcome_column") or payload.get("target_column") or None

        report = build_quality_report(
            df,
            protected,
            outcome_column=outcome,
            unavailable_attributes=unavailable,
        )
        return TaskResult.ok(_TASK_TYPE, data=report).to_dict()
    except Exception as exc:  # pragma: no cover -- surfaced verbatim
        return TaskResult.fail(_TASK_TYPE, f"Data validation failed: {exc}").to_dict()


def main() -> int:
    """``python -m vfairness.operations.cicd.task_handlers vfairness_data_validation``."""
    task_type = sys.argv[1] if len(sys.argv) > 1 else "vfairness_data_validation"
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
    result = handle_data_validation(payload)
    json.dump(result, sys.stdout, default=str)
    # FAIL CLOSED at the exit code, the only thing a shell or a CI step reads.
    # BGL5 AUDIT, measured 2026-09-27:
    #   $ python -m vfairness.operations.cicd.task_handlers \
    #         vfairness_data_validation < payload.json
    #   {"schema_version": "1.0", "task_type": "vfairness_data_validation",
    #    "success": false, "error": "None of the protected attributes are present
    #    in the data, so group fairness cannot be validated."}
    #   exit code: 0
    # After this change the same run exits 1, and a successful validation (the
    # 400-row frame with 'gender' present) still exits 0 with the same stdout.
    #
    # Bad JSON and an unknown task type already returned 1 above, so the exit
    # code WAS being used to signal failure, inconsistently: an infrastructure
    # failure was 1 and a check that could not run was 0. The module header names
    # CI jobs as a consumer, and the sibling precommit.main was fixed for exactly
    # this doctrine ("a check that never ran must not report success").
    #
    # stdout is unchanged, so a task consumer that parses the envelope keeps
    # reading `success` and `error` as before; only the code a shell sees moves.
    return 0 if result.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
