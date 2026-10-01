"""Task handlers for the validity axis (``vfairness_validity_run``).

Mirrors the ``operations/causal/task_handlers.py`` pattern: a ``_HANDLERS`` dict
maps the task type to ``handle_*(payload) -> TaskResult.to_dict()``, invoked as
``python -m vfairness.operations.validity.task_handlers vfairness_validity_run``
with the JSON payload piped over stdin and the envelope printed on stdout.

FAIL-CLOSED: with no groundedness scorer wired (the case until VA-10/VA-21 land),
the handler still returns ``success=True`` but every record is marked
``available=False`` and the aggregate reports ``available=False`` with a warning.
It never emits a fabricated score, and the platform keeps such a result in the
``metricsDeferred`` channel (feedsGrade stays false).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from vfairness.net.egress import SSRFError, validate_endpoint
from vfairness.result import TaskResult
from vfairness.validity.aggregate import aggregate_validity
from vfairness.validity.groundedness import GroundednessResult, GroundednessScorer
from vfairness.validity.judge import LlmGroundednessJudge

TASK_TYPE = "vfairness_validity_run"

_ALLOW_PRIVATE_ENV = "VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE"


def _endpoint_allowed(url: str) -> bool:
    """SSRF guard for the judge endpoint before any request is made.

    The endpoint can arrive in the (potentially attacker-influenceable) task payload,
    and the judge both POSTs to it AND parses its reply as the sealed groundedness
    verdict, so an unguarded endpoint is a server-side request forgery that also
    controls the score. Route it through the same egress guard the LLM proxy uses
    (``net.egress.validate_endpoint``): a local judge (Ollama on 127.0.0.1) is the
    first-class operator case, so loopback is permitted, but link-local / cloud-metadata
    (169.254.169.254) and RFC1918 internal services are refused. The scheme-only check
    in ``LlmGroundednessJudge.__init__`` is not an egress guard.

    An operator who runs the judge on a private LAN address and trusts the payload
    channel can opt in with ``VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE`` (skips the guard),
    matching the deliberate-opt-in posture ``LLMApiProxy(allow_loopback=...)`` uses.
    """
    if os.environ.get(_ALLOW_PRIVATE_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        validate_endpoint(url, allow_http=True, allow_loopback=True)
        return True
    except SSRFError:
        return False


def _answer_of(rec: dict) -> str:
    """The answer a record carries, or "" when it carries none.

    Absence has six reachable spellings and `str()` turns every one of them into a
    word: None becomes "None", float nan becomes "nan", pd.NA becomes "<NA>",
    pd.NaT becomes "NaT". Each is truthy and non-blank, so each walks past an
    empty-answer guard and gets scored. Returning "" hands the scorer the one
    shape it already refuses honestly.
    """
    value = rec.get("answer")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import pandas as _pd

        if _pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    # The renderings str() MINTS from a non-string absent value. This is reached
    # only for non-strings, because a str is returned above it untouched, and that
    # ordering is deliberate: a caller who literally sent "nan" or "None" sent a
    # string, and manufacturing absence out of their data is the mirror of the
    # defect being fixed here. The sibling decision in this campaign went the same
    # way, keeping the literal 'None' as a demographic group and disclosing it,
    # because 'None' is a real category for some attributes. So this set exists for
    # np.ma.masked, which renders '--', and for anything else whose str() collides
    # with an absence token. Matched EXACTLY, never by substring.
    if text in {"None", "nan", "NaN", "<NA>", "NaT", "--"}:
        return ""
    return text


def _contexts_of(record: Dict[str, Any]) -> List[str]:
    raw = record.get("retrieved_context") or record.get("contexts") or []
    out: List[str] = []
    for c in raw:
        if isinstance(c, str):
            if c.strip():
                out.append(c)
        elif isinstance(c, dict):
            text = str(c.get("text", "")).strip()
            if text:
                out.append(text)
    return out


def _build_scorer(payload: Dict[str, Any]) -> GroundednessScorer:
    """Build the scorer from config, fail-closed.

    The interim judge (VA-10, Mistral Small 3.2) is built from a ``judge`` block in
    the payload or the ``VFAIRNESS_VALIDITY_JUDGE_*`` env vars: URL, MODEL, and
    optionally FORMAT / TOKEN. With neither an endpoint nor a model configured, or
    on any construction error, this returns a scorer with no rung, so ``score()``
    refuses and nothing feeds the grade. The owned detector (VA-21 sidecar) is
    wired here in a follow-up.
    """
    jc = payload.get("judge") or {}
    url = jc.get("endpoint_url") or os.environ.get("VFAIRNESS_VALIDITY_JUDGE_URL")
    model = jc.get("model_name") or os.environ.get("VFAIRNESS_VALIDITY_JUDGE_MODEL")
    if url and model:
        if not _endpoint_allowed(url):
            # SSRF-refused (loopback ok, but link-local/metadata/RFC1918 refused). Fail
            # closed: no rung, so score() refuses and nothing feeds the grade or seal.
            return GroundednessScorer(judge=None, sidecar=None)
        # Guard every redirect hop unless the operator explicitly opted into private
        # targets (the same opt-in that relaxes the up-front _endpoint_allowed check).
        # Without the opt-in a payload endpoint that 302-redirects to a metadata /
        # RFC1918 host is refused mid-flight, not just on the first hop.
        allow_private = os.environ.get(_ALLOW_PRIVATE_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            judge = LlmGroundednessJudge(
                endpoint_url=url,
                model_name=model,
                api_format=jc.get("api_format")
                or os.environ.get("VFAIRNESS_VALIDITY_JUDGE_FORMAT", "openai"),
                auth_token=jc.get("auth_token") or os.environ.get("VFAIRNESS_VALIDITY_JUDGE_TOKEN"),
                guard_egress=not allow_private,
            )
            return GroundednessScorer(judge=judge, sidecar=None)
        except Exception:
            pass  # bad config -> fail closed (no rung)
    return GroundednessScorer(judge=None, sidecar=None)


def handle_validity_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    records = payload.get("records") or payload.get("eval_set") or []
    if not isinstance(records, list):
        return TaskResult.fail(
            TASK_TYPE, "payload.records must be a list of eval-set records"
        ).to_dict()

    default_language = str(payload.get("language", "unknown"))
    scorer = _build_scorer(payload)

    # G12, 2026-09-30. NON-DICT RECORDS WERE DROPPED SILENTLY, and a payload of
    # NOTHING BUT them was accepted as a complete run. ``records`` not being a
    # list is refused above; a list whose ITEMS are not records was the partial
    # (and total) loss beside it. Measured before this change:
    #   handle_validity_run({"records": ["a", "b", "c"]})
    #     -> success=True, record_count=0, aggregate n=0, warnings: only the
    #        "no scorer wired" line. Three records in, nothing measured, and
    #        nothing in the envelope said so.
    #   handle_validity_run({"records": [rec, "junk", 5, None, rec]})
    #     -> record_count=2, and the 3 dropped records appeared nowhere; with a
    #        judge wired, the aggregate mean would be published over 2 of 5
    #        records as the batch result.
    # `record_count` is the number SCORED and is asserted by the existing suite,
    # so it keeps its meaning; the supplied and skipped counts are added beside
    # it, and a skip is warned about, because a count a reader has to subtract
    # two other counts to find is not a disclosure.
    results: List[GroundednessResult] = []
    n_skipped = 0
    for rec in records:
        if not isinstance(rec, dict):
            n_skipped += 1
            continue
        results.append(
            scorer.score(
                # `.get(key, default)` DOES NOT FIRE FOR A PRESENT KEY HOLDING None,
                # so `{"answer": null}`, which is what a JSON payload with no answer
                # looks like, became the five-character string "None". That is truthy
                # and non-blank, so the scorer's own empty-answer guard passed it
                # through and scored a hallucination rate against a word the model
                # never said. `str(x)` on an absent value MINTS CONTENT; this is the
                # same door that put 'None' into a demographic group holding a fifth
                # of a population and produced a redlining finding against a place
                # that is not a place. Found by the G10 agent outside its own package
                # boundary and reported rather than reached into, 2026-09-30.
                #
                # The empty string is the right substitute because the scorer already
                # treats it as an absent answer and refuses; what was wrong was
                # manufacturing a word. All six absence spellings are handled by
                # _answer_of, which is shared with nothing else on purpose.
                _answer_of(rec),
                _contexts_of(rec),
                question=rec.get("prompt"),
                gold=rec.get("reference_answer"),
                language=str(rec.get("language", default_language)),
            )
        )

    aggregate = aggregate_validity(results)
    warnings_list: List[str] = []
    if not scorer.available:
        warnings_list.append(
            "No groundedness scorer wired yet (VA-10/VA-21 pending); results are "
            "configured but NOT measured. Fail-closed: nothing feeds the grade or seal."
        )
    if n_skipped:
        warnings_list.append(
            f"{n_skipped} of {len(records)} payload.records entries are not records (not a "
            f"JSON object) and were NOT SCORED. They are excluded from record_count, from "
            f"the per-record list and from the aggregate, so this run covers "
            f"{len(results)} of {len(records)} supplied records and its aggregate is NOT a "
            f"measurement of the whole batch. See records_supplied / records_skipped."
        )

    return TaskResult.ok(
        TASK_TYPE,
        data={
            "records": [r.to_dict() for r in results],
            "aggregate": aggregate,
            "scorer_available": scorer.available,
            "record_count": len(results),
            # The denominator the aggregate is NOT over, stated outright.
            "records_supplied": len(records),
            "records_skipped": n_skipped,
        },
        warnings=warnings_list or None,
    ).to_dict()


_HANDLERS = {TASK_TYPE: handle_validity_run}


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(
            json.dumps(
                TaskResult.fail("vfairness_validity", "Usage: python -m ... <task_type>").to_dict()
            )
        )
        sys.exit(2)

    task_type = sys.argv[1]
    handler = _HANDLERS.get(task_type)
    if handler is None:
        print(json.dumps(TaskResult.fail(task_type, f"Unknown task_type: {task_type}").to_dict()))
        sys.exit(2)

    payload_text = sys.stdin.read() or "{}"
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        print(json.dumps(TaskResult.fail(task_type, f"Invalid JSON payload: {exc}").to_dict()))
        sys.exit(2)

    print(json.dumps(handler(payload), default=str))


if __name__ == "__main__":
    main()
