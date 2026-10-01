"""Pulse standard trace ingestion (Pulse close plan G-10, with the G-43
memory-event fields reserved in the v1 contract).

The 2026 agent ecosystem emits OpenTelemetry GenAI spans (``gen_ai.*``
attributes, from LangChain / CrewAI / AutoGen instrumentation) and
Langfuse / LangSmith style JSON exports. Pulse's agent probe consumes a
flat per-episode table (one row per trace with a demographic group and
the tool/action taken). This module deterministically flattens the two
raw export shapes into that table:

- ``parse_trace_export(records)``: list of span/observation dicts (or a
  wrapped container) to the per-episode DataFrame. Liberal on input
  (flat or nested attributes, OTLP keyed-list attributes, camelCase or
  snake_case ids), honest on output (unknown fields become NaN, never
  fabricated values).
- ``maybe_flatten_spans(df)``: recognizes a DataFrame that is a raw
  span/observation export (rather than the per-episode table),
  reconstructs the records and delegates to ``parse_trace_export``.
  Returns None whenever the frame is not span-shaped, so callers proceed
  exactly as before.
- ``ingest_and_run_pulse(df, inputs)``: the package-level ``run_pulse``
  front door. For agent-declared (or undeclared) runs it flattens raw
  span exports BEFORE the orchestrator's column detection and discloses
  the ingestion form to the agent probe via ``inputs._trace_ingestion``;
  every other frame passes through to
  ``vfairness.operations.pulse.orchestrator.run_pulse`` untouched, which
  stays the single source of truth for the analysis.

Everything here is deterministic parsing; no LLM is involved. The
memory-event columns (``memory_reads`` / ``memory_writes``) are parsed
when memory-ish spans are present and left NaN otherwise; the
memory-contamination SCREEN itself (G-43) is not implemented yet. See
TRACE_CONTRACT.md next to this module for the published schema.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class MalformedTraceExportWarning(UserWarning):
    """A cell opened with { or [ and did not parse as JSON.

    READINESS-6, 2026-09-10. ``_maybe_json`` returns the ORIGINAL string on any
    parse failure, and ``maybe_flatten_spans`` returns None on anything it
    cannot flatten, so "this is not a trace export" and "this IS a trace export
    and your exporter truncated it" were the same silent answer.

    MEASURED: an intact span export flattened to 8 per-episode rows; the same
    export with each JSON cell truncated mid-object returned None with no
    warning, and ``ingest_and_run_pulse`` then ran the full pulse on the raw
    one-column frame. The user gets a REPORT where they should get a refusal. A
    CSV cell cut short by an exporter is the ordinary way this happens.

    A warning, deliberately, and never an exception: raising would break the
    legitimate pass-through that lets a genuine non-trace frame reach the
    orchestrator untouched.
    """


# The published per-episode column set (v1). Always emitted in full so
# the contract is stable; missing information is NaN, never invented.
# memory_read_text / memory_write_text are the v1 additive enrichment
# for the term-level contamination screen: when a memory-ish span
# carries a content-like attribute, its text is retained per episode
# (bounded), used transiently by the analysis and never persisted.
EPISODE_COLUMNS = (
    "trace_id",
    "group",
    "tool",
    "tool_calls",
    "route",
    "steps",
    "outcome",
    "timestamp",
    "memory_reads",
    "memory_writes",
    "memory_read_text",
    "memory_write_text",
)

# Content-like attribute keys a memory-ish span may carry its payload
# under (exact key or dotted suffix, same lookup as the other carriers),
# plus the Langfuse top-level input/output fields handled inline.
_MEMORY_TEXT_KEYS = (
    "gen_ai.prompt",
    "gen_ai.completion",
    "content",
    "memory.content",
    "input",
    "output",
    "text",
    "value",
)
# Per-episode, per-direction retention bound. Enough for a term screen;
# small enough that a chatty memory layer cannot balloon the frame.
_MEMORY_TEXT_CAP = 4000

# Demographic-group carriers, looked up in span attributes, Langfuse
# metadata and top-level record fields (exact key or dotted suffix).
GROUP_KEYS = ("group", "demographic", "persona", "segment", "user_group")

_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name", "tool_name", "tool")
_TOOL_OPERATIONS = ("execute_tool", "invoke_tool", "tool", "tool_call")
_OPERATION_KEYS = ("gen_ai.operation.name", "operation.name", "operation")
_ROUTE_KEYS = (
    "gen_ai.agent.delegate_to",
    "agent.delegate_to",
    "delegate_to",
    "gen_ai.handoff.to",
    "handoff_to",
    "handoff",
    "delegation_target",
    "route_to",
)
_START_KEYS = (
    "start_time_unix_nano",
    "starttimeunixnano",
    "start_time",
    "starttime",
    "timestamp",
    "time",
    "ts",
    "created_at",
    "createdat",
    "started_at",
    "startedat",
)
_TRACE_ID_KEYS = ("trace_id", "traceid", "trace.id")
_SPAN_ID_COLUMNS = ("span_id", "spanid", "observation_id", "observationid")
_PAYLOAD_COLUMNS = ("attributes", "observations", "metadata")
_GROUP_COLUMN_NAMES = ("group", "demographic", "persona", "segment", "cohort", "user_group")
_TOOL_COLUMN_NAMES = ("tool", "tool_name", "action", "selected_tool", "agent_action")
_LANGFUSE_TYPES = {
    "SPAN",
    "GENERATION",
    "EVENT",
    "TOOL",
    "AGENT",
    "CHAIN",
    "RETRIEVER",
    "EMBEDDING",
}
# G-43 groundwork: memory-ish operation/span names. Write verbs are
# checked first so "memory.upsert" is never misread as a read.
_MEMORY_WRITE_VERBS = (
    "write",
    "put",
    "store",
    "save",
    "add",
    "update",
    "insert",
    "upsert",
    "append",
    "set",
    "persist",
)
_MEMORY_READ_VERBS = ("read", "get", "retrieve", "load", "recall", "search", "query", "fetch")
# Top-level record fields that are containers, not attribute scalars.
_STRUCTURAL_KEYS = {
    "attributes",
    "metadata",
    "observations",
    "events",
    "links",
    "resource",
    "scope",
    "spans",
    "status",
}
_TRUTHY = {"true", "1", "yes", "success", "ok", "passed", "resolved"}
_FALSY = {"false", "0", "no", "fail", "failure", "error", "unresolved"}

_AGENT_SOURCE_KINDS = ("", "agent", "agent_traces", "traces", "multiagent", "agentic")


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _maybe_json_checked(value: str) -> Tuple[Any, bool]:
    """Parse a JSON-looking string. Returns ``(value, looked_like_json_and_failed)``.

    The second element is THE distinction this module was missing: a cell that
    never looked like JSON and a cell that opened with ``{`` and could not be
    parsed are different facts about the upload, and only one of them is the
    user's exporter truncating their data. See MalformedTraceExportWarning.
    """
    text = value.strip()
    if text[:1] not in ("{", "["):
        return value, False
    try:
        return json.loads(text), False
    except Exception:  # noqa: BLE001
        return value, True


def _maybe_json(value: str):
    """Parse a JSON-looking string; return the original on any failure."""
    parsed, _malformed = _maybe_json_checked(value)
    return parsed


def _unwrap_otlp_value(v: Any) -> Any:
    """OTLP-JSON wraps attribute values as {"stringValue": ...} etc."""
    if isinstance(v, dict):
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if key in v:
                return v[key]
        return None
    return v


def _flatten_attrs(record: Dict[str, Any]) -> Dict[str, Any]:
    """Merge span attributes, Langfuse metadata and top-level scalar
    fields into one flat lowercase-keyed dict. Attribute/metadata values
    take precedence over top-level fields of the same name."""
    attrs: Dict[str, Any] = {}

    def _absorb(obj: Any) -> None:
        if isinstance(obj, str):
            obj = _maybe_json(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                v2 = _unwrap_otlp_value(v)
                if isinstance(v2, (str, int, float, bool)):
                    attrs.setdefault(str(k).strip().lower(), v2)
        elif isinstance(obj, list):
            # OTLP keyed-list form: [{"key": ..., "value": {...}}, ...]
            for item in obj:
                if isinstance(item, dict) and "key" in item:
                    v2 = _unwrap_otlp_value(item.get("value"))
                    if isinstance(v2, (str, int, float, bool)):
                        attrs.setdefault(str(item["key"]).strip().lower(), v2)

    _absorb(record.get("attributes"))
    _absorb(record.get("metadata"))
    for k, v in record.items():
        kl = str(k).strip().lower()
        if kl in _STRUCTURAL_KEYS:
            continue
        if isinstance(v, (str, int, float, bool)):
            attrs.setdefault(kl, v)
    return attrs


def _serialised_absence_strings() -> frozenset:
    """The strings that mean "no value" once absence has been through a file.

    PANDAS' OWN DEFAULT LIST, not a new one. ``pd.read_csv`` converts every
    member of ``STR_NA_VALUES`` to NaN on every CSV this library reads, so
    '<NA>', 'None', 'nan', 'NaN', 'null', 'NA' and 'N/A' are ALREADY absence on
    the tabular path. The trace path was the outlier: a span export is a JSON
    string inside a cell, ``json.loads`` hands back the string unchanged, and
    nothing here agreed with the reader one layer up. Reusing the list makes the
    two paths answer one question one way instead of two.

    'NaT' is added because it is the one ``str()`` of a pandas absence sentinel
    that pandas' own reader does not list, and it is how ``pd.NaT`` survives
    ``json.dumps(..., default=str)``, which is how a trace exporter serialises a
    dataframe-backed attribute. Measured: that door published an adverse opinion
    against a group named 'NaT'. Nobody names a demographic group 'NaT'.

    A literal category a person chose, such as 'missing' or 'unknown', is NOT
    here and must not be: this library already refused to let its own
    ``__missing__`` sentinel swallow a real level spelled "missing".
    """
    base = {"NaT"}
    try:
        from pandas._libs.parsers import STR_NA_VALUES

        base |= set(STR_NA_VALUES)
    except Exception:  # noqa: BLE001
        # A private pandas path. If it ever moves, fall back to the members that
        # matter for THIS defect rather than silently checking nothing.
        base |= {"", "<NA>", "None", "nan", "NaN", "null", "NULL", "NA", "N/A", "n/a"}
    return frozenset(base)


_SERIALISED_ABSENCE = _serialised_absence_strings()


def _attr_is_absent(v: Any) -> bool:
    """Is this attribute value NO VALUE, by every door absence arrives through?

    ``None`` WAS THE ONLY DOOR (grade wave G06, 2026-09-30). The test was
    ``v is not None and str(v).strip() != ""``, and ``str()`` on an absent value
    MINTS CONTENT: ``str(float('nan'))`` is the four-character word 'nan',
    ``str(pd.NA)`` is '<NA>' and ``str(pd.NaT)`` is 'NaT', none of which is the
    empty string, so all three were returned as a value and
    ``_add_event``/``_absorb_header`` wrote ``b["group"] = str(g)``.

    Measured on this repo before the fix, 60 trace episodes, half labelled "A"
    and half whose ``user.group`` attribute is absent, the absent half failing
    more often (a real trajectory gap), through ``ingest_and_run_pulse``:

        float('nan') -> "Adverse opinion: material fairness defects make this
                         system unfit to deploy as-is. Trajectory gap on
                         'outcome': A averages 0.80 vs 0.30 for nan (outcome
                         rate; permutation test, BH-adjusted p 0.001998,
                         Cohen's d 1.14)."
        pd.NA        -> the same sentence, "for <NA>"
        pd.NaT       -> the same sentence, "for NaT"

    byte-for-byte the sentence a genuine second group "B" produces on the same
    data. An adverse opinion is a claim about a protected class, and "this
    episode carries no record of the acting group" is not one; a reader cannot
    tell the three apart from the real finding, and the group named 'nan' is the
    one the defect is attributed to.

    The rule is the library's ``pd.isna``, the same one
    ``_validation.handle_missing_values`` uses for
    ``missing_strategy='exclude'``, so this does not become another disagreeing
    copy. A whitespace-only string is normalised into the same absence, which is
    what the old test already did for ``""``.

    THE LITERAL STRING 'None' IS NOT ABSENCE and is deliberately left alone: it
    is indistinguishable from a group somebody named, and this library already
    made that call in the same direction when it spelled its own sentinel
    ``__missing__`` so it could never swallow a real level called "missing".

    ``pd.isna`` returns an ARRAY for a list or ndarray, so the scalar check is
    guarded: ``if pd.isna(v)`` on an array raises ValueError, which inside
    ``parse_trace_export`` would have turned a list-valued attribute into a
    skipped record.
    """
    if v is None:
        return True
    if isinstance(v, str):
        # A whitespace-only string, and the SERIALISED reprs of absence: a span
        # export is JSON inside a cell, so ``pd.NA`` reaches here as the text
        # '<NA>' and no test of the object can see it. See
        # _serialised_absence_strings for whose list this is and why 'missing'
        # is deliberately not on it.
        return not v.strip() or v.strip() in _SERIALISED_ABSENCE
    if isinstance(v, (list, tuple, set, dict, np.ndarray, pd.Series, pd.DataFrame)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _lookup(attrs: Dict[str, Any], names: Sequence[str]) -> Optional[Any]:
    """Exact-key lookup first, then a dotted-suffix match (so
    "metadata.group" or "gen_ai.tool.name" resolve). Deliberately NOT an
    underscore-suffix match: "age_group" must never satisfy "group".

    An absent value is SKIPPED rather than returned, by every door absence
    arrives through; see :func:`_attr_is_absent` for the three that used to mint
    a group label and the measured before-state. Skipping rather than returning
    also means a record carrying both an absent ``group`` and a present
    ``metadata.group`` now resolves to the real one instead of to 'nan'.
    """
    for n in names:
        v = attrs.get(n)
        if not _attr_is_absent(v):
            return v
    for n in names:
        suffix = "." + n
        for k in sorted(attrs):
            if k.endswith(suffix):
                v = attrs[k]
                if not _attr_is_absent(v):
                    return v
    return None


def _to_epoch_seconds(v: Any) -> Optional[float]:
    """Normalize numeric epochs (nano/micro/milli/second) and ISO-8601
    strings to float seconds; None when unparseable. Small integers
    (step indices) pass through unchanged as an ordering value."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        if not np.isfinite(x):
            return None
        if abs(x) >= 1e17:
            return x / 1e9
        if abs(x) >= 1e14:
            return x / 1e6
        if abs(x) >= 1e11:
            return x / 1e3
        return x
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return _to_epoch_seconds(float(s))
        except ValueError:
            pass
        ts = _safe(lambda: pd.to_datetime(s, utc=True), None)
        if ts is None or pd.isna(ts):
            return None
        return float(ts.timestamp())
    return None


def _event_tool(
    attrs: Dict[str, Any], name: str, op: Optional[Any], rec_type: str
) -> Optional[str]:
    """The tool name of one span/observation, or None when the event is
    not a tool invocation. Sources, in order: an explicit tool-name
    attribute; a tool-execution operation name (the span name then
    carries the tool); a Langfuse observation of type TOOL."""
    t = _lookup(attrs, _TOOL_NAME_KEYS)
    if t is not None:
        return str(t)
    op_l = str(op).strip().lower() if op is not None else ""
    if op_l in _TOOL_OPERATIONS and name:
        parts = name.split()
        return parts[-1] if parts else name
    if rec_type == "TOOL" and name:
        return name
    return None


def _memory_kind(name: str, op: Optional[Any]) -> Optional[str]:
    """Classify a memory-ish event as 'read' or 'write' from its span
    name / operation name; None when the event is not memory-ish or the
    verb is ambiguous (ambiguous events are counted toward neither)."""
    text = " ".join(str(x).lower() for x in (name, op) if x)
    if "memory" not in text and "mem0" not in text:
        return None
    for verb in _MEMORY_WRITE_VERBS:
        if verb in text:
            return "write"
    for verb in _MEMORY_READ_VERBS:
        if verb in text:
            return "read"
    return None


def _event_status(record: Dict[str, Any], attrs: Dict[str, Any]) -> Optional[str]:
    """'success' / 'failure' when derivable from explicit success/outcome
    attributes, the OTel span status, or the Langfuse level; else None
    (never guessed)."""
    v = _lookup(attrs, ("success", "outcome"))
    if isinstance(v, bool):
        return "success" if v else "failure"
    if v is not None:
        s = str(v).strip().lower()
        if s in _TRUTHY:
            return "success"
        if s in _FALSY:
            return "failure"
    status = record.get("status")
    code = status.get("code") if isinstance(status, dict) else status
    if code is None:
        code = attrs.get("otel.status_code")
    if code is not None and not isinstance(code, (dict, list)):
        cs = str(code).strip().upper()
        if cs in ("2", "STATUS_CODE_ERROR", "ERROR"):
            return "failure"
        if cs in ("1", "STATUS_CODE_OK", "OK"):
            return "success"
    if str(record.get("level") or "").strip().upper() == "ERROR":
        return "failure"
    return None


def _record_trace_id(record: Dict[str, Any]) -> Optional[str]:
    for k in record:
        if str(k).strip().lower() in _TRACE_ID_KEYS:
            v = record[k]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            s = str(v).strip()
            if s:
                return s
    return None


def _record_plain_id(record: Dict[str, Any]) -> Optional[str]:
    for k in record:
        if str(k).strip().lower() == "id":
            v = record[k]
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def _looks_like_trace_header(record: Dict[str, Any]) -> bool:
    """A Langfuse-style trace object: has an id and metadata/tags but no
    span/observation markers. Carries trace-level group metadata; it is
    not itself a step."""
    low = {str(k).strip().lower() for k in record}
    if low & {"span_id", "spanid", "type", "parentobservationid"}:
        return False
    return "id" in low and bool(low & {"metadata", "tags"})


def _normalize_container(records: Any) -> List[Any]:
    """Accept a plain list of dicts, or a wrapped container: OTLP-JSON
    resourceSpans/scopeSpans nesting, or {spans|observations|traces|
    records|data|batch: [...]}."""
    if isinstance(records, dict):
        for key in ("resourceSpans", "resource_spans"):
            if isinstance(records.get(key), list):
                spans: List[Any] = []
                for rs in records[key]:
                    if not isinstance(rs, dict):
                        continue
                    for ss_key in ("scopeSpans", "scope_spans", "instrumentationLibrarySpans"):
                        for ss in rs.get(ss_key) or []:
                            if isinstance(ss, dict):
                                spans.extend(ss.get("spans") or [])
                return spans
        for key in ("spans", "observations", "traces", "records", "data", "batch"):
            if isinstance(records.get(key), list):
                return list(records[key])
        return [records]
    if isinstance(records, list):
        return list(records)
    return []


def _detect_form(records: Sequence[Any]) -> str:
    """'otel_spans' vs 'langfuse' by marker votes over the records.
    Ties resolve to 'otel_spans' (the more standardized shape)."""
    otel = 0
    langfuse = 0
    for rec in list(records)[:500]:
        if not isinstance(rec, dict):
            continue
        low = {str(k).strip().lower() for k in rec}
        if low & {"span_id", "spanid"}:
            otel += 2
        if "trace_id" in low:
            otel += 1
        if isinstance(rec.get("attributes"), (dict, list)):
            otel += 2
        if any(str(k).lower().startswith("gen_ai.") for k in rec):
            otel += 2
        a = rec.get("attributes")
        if isinstance(a, dict) and any(str(k).lower().startswith("gen_ai.") for k in a):
            otel += 1
        if isinstance(rec.get("observations"), list):
            langfuse += 3
        if "metadata" in low:
            langfuse += 2
        if str(rec.get("type") or "").strip().upper() in _LANGFUSE_TYPES:
            langfuse += 2
        if "traceid" in low and "spanid" not in low:
            langfuse += 1
    return "langfuse" if langfuse > otel else "otel_spans"


def parse_trace_export(records: Any) -> pd.DataFrame:
    """Flatten a raw span/observation export into the per-episode table
    (one row per trace_id, columns ``EPISODE_COLUMNS``).

    Accepts a list of dicts in either shape (OTLP-JSON-ish spans or
    Langfuse-ish trace/observation objects; nested observations under a
    trace object are handled), or a wrapped container. Deterministic:
    no LLM, no sampling. Missing information is NaN, never fabricated;
    per the G-43 reservation, memory_reads/memory_writes are counted
    when memory-ish spans exist anywhere in the export and left NaN
    otherwise. Records that are not objects or carry no trace id are
    skipped and counted in the ingestion notes.
    """
    recs = _normalize_container(records)
    form = _detect_form(recs)
    buckets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    skipped = 0
    events = 0
    any_memory = False

    def _bucket(tid: str) -> Dict[str, Any]:
        b = buckets.get(tid)
        if b is None:
            b = {
                "group": None,
                "route": None,
                "steps": 0,
                "tools": [],
                "starts": [],
                "mem_r": 0,
                "mem_w": 0,
                "mem_r_txt": [],
                "mem_w_txt": [],
                "mem_r_len": 0,
                "mem_w_len": 0,
                "err": 0,
                "ok": 0,
            }
            buckets[tid] = b
            order.append(tid)
        return b

    def _memory_text(rec: Dict[str, Any], attrs: Dict[str, Any]) -> str:
        """Best-effort payload text of a memory-ish span: the first
        content-like attribute, else the Langfuse top-level
        input/output. Empty when the span carries no content."""
        val = _lookup(attrs, _MEMORY_TEXT_KEYS)
        if val is None:
            for key in ("input", "output"):
                v = rec.get(key)
                if v is not None:
                    val = v
                    break
        return "" if val is None else str(val)

    def _retain_memory_text(b: Dict[str, Any], direction: str, text: str) -> None:
        if not text:
            return
        key_txt = "mem_r_txt" if direction == "read" else "mem_w_txt"
        key_len = "mem_r_len" if direction == "read" else "mem_w_len"
        room = _MEMORY_TEXT_CAP - b[key_len]
        if room <= 0:
            return
        clipped = text[:room]
        b[key_txt].append(clipped)
        b[key_len] += len(clipped) + 1  # +1 for the join separator

    def _absorb_header(b: Dict[str, Any], rec: Dict[str, Any]) -> None:
        attrs = _flatten_attrs(rec)
        if b["group"] is None:
            g = _lookup(attrs, GROUP_KEYS)
            if g is not None:
                b["group"] = str(g)
        if b["route"] is None:
            r = _lookup(attrs, _ROUTE_KEYS)
            if r is not None:
                b["route"] = str(r)

    def _add_event(b: Dict[str, Any], rec: Dict[str, Any]) -> None:
        nonlocal events, any_memory
        events += 1
        attrs = _flatten_attrs(rec)
        b["steps"] += 1
        name = str(rec.get("name") or "")
        op = _lookup(attrs, _OPERATION_KEYS)
        rec_type = str(rec.get("type") or "").strip().upper()
        start = _to_epoch_seconds(_lookup(attrs, _START_KEYS))
        if start is not None:
            b["starts"].append(start)
        tool = _event_tool(attrs, name, op, rec_type)
        if tool is not None:
            b["tools"].append((start if start is not None else float("inf"), events, str(tool)))
        if b["group"] is None:
            g = _lookup(attrs, GROUP_KEYS)
            if g is not None:
                b["group"] = str(g)
        if b["route"] is None:
            r = _lookup(attrs, _ROUTE_KEYS)
            if r is not None:
                b["route"] = str(r)
        mem = _memory_kind(name, op)
        if mem == "read":
            b["mem_r"] += 1
            any_memory = True
            _retain_memory_text(b, "read", _memory_text(rec, attrs))
        elif mem == "write":
            b["mem_w"] += 1
            any_memory = True
            _retain_memory_text(b, "write", _memory_text(rec, attrs))
        status = _event_status(rec, attrs)
        if status == "failure":
            b["err"] += 1
        elif status == "success":
            b["ok"] += 1

    for rec in recs:
        if not isinstance(rec, dict):
            skipped += 1
            continue
        obs = rec.get("observations")
        if isinstance(obs, list):
            tid = _record_trace_id(rec) or _record_plain_id(rec)
            if tid is None:
                skipped += 1
                continue
            b = _bucket(tid)
            _absorb_header(b, rec)
            for o in obs:
                if isinstance(o, dict):
                    _add_event(b, o)
                else:
                    skipped += 1
            continue
        tid = _record_trace_id(rec)
        if tid is not None:
            _add_event(_bucket(tid), rec)
            continue
        if _looks_like_trace_header(rec):
            tid = _record_plain_id(rec)
            if tid is not None:
                _absorb_header(_bucket(tid), rec)
                continue
        skipped += 1

    rows: List[Dict[str, Any]] = []
    for tid in order:
        b = buckets[tid]
        tools_sorted = sorted(b["tools"])
        has_events = b["steps"] > 0
        rows.append(
            {
                "trace_id": tid,
                "group": b["group"] if b["group"] is not None else np.nan,
                "tool": tools_sorted[0][2] if tools_sorted else np.nan,
                "tool_calls": (float(len(tools_sorted)) if has_events else np.nan),
                "route": b["route"] if b["route"] is not None else np.nan,
                "steps": float(b["steps"]) if has_events else np.nan,
                "outcome": ("failure" if b["err"] > 0 else ("success" if b["ok"] > 0 else np.nan)),
                "timestamp": (float(min(b["starts"])) if b["starts"] else np.nan),
                "memory_reads": (float(b["mem_r"]) if any_memory and has_events else np.nan),
                "memory_writes": (float(b["mem_w"]) if any_memory and has_events else np.nan),
                # Term-level enrichment: empty string = memory events were
                # counted but carried no content-like payload; NaN = the
                # export carried no memory events at all.
                "memory_read_text": (
                    "\n".join(b["mem_r_txt"]) if any_memory and has_events else np.nan
                ),
                "memory_write_text": (
                    "\n".join(b["mem_w_txt"]) if any_memory and has_events else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    memory_note = (
        "memory events present"
        if any_memory
        else "no memory events; memory_reads/memory_writes left NaN"
    )
    out.attrs["ingestion"] = {
        "form": form,
        "notes": (
            "Flattened {e} span/observation event(s) across {t} "
            "trace(s) from a {f} export; {s} record(s) skipped "
            "(not an object or no trace id); {m}.".format(
                e=events, t=len(order), f=form, s=skipped, m=memory_note
            )
        ),
    }
    return out


def maybe_flatten_spans(
    df: Any, diagnostics: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """Flatten a DataFrame that is a raw span/observation export into
    the per-episode table; None when the frame is not span-shaped (the
    caller proceeds exactly as before). Never raises.

    ``diagnostics``, when supplied, is filled in with what this pass saw:
    ``malformedJsonCells`` (cells that opened with { or [ and did not parse),
    ``cellsScanned``, and a plain ``note``. That is the DISCLOSURE half of the
    fix described on MalformedTraceExportWarning: the return value stays None
    for a malformed export so the pass-through contract is unchanged, and the
    caller can now tell the two Nones apart.
    """
    diag: Dict[str, Any] = {} if diagnostics is None else diagnostics
    flat = _safe(lambda: _maybe_flatten(df, diag), None)
    n_bad = int(diag.get("malformedJsonCells") or 0)
    if n_bad:
        note = (
            "{0} of {1} cell(s) in this upload begin with '{{' or '[' and are not "
            "valid JSON, which is what a trace export truncated by its exporter "
            "looks like. {2}".format(
                n_bad,
                int(diag.get("cellsScanned") or 0),
                (
                    "The frame was still flattened; rows from the unparseable cells "
                    "are missing from the analysis."
                    if flat is not None
                    else "The frame could NOT be read as a trace export, so it was passed "
                    "through to the ordinary tabular analysis unchanged. If this was "
                    "meant to be an agent trace export, re-export it: the report you "
                    "get will not be about your traces."
                ),
            )
        )
        diag["note"] = note
        warnings.warn(note, MalformedTraceExportWarning, stacklevel=2)
    return flat


def _maybe_flatten(df: Any, diag: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    if diag is None:
        diag = {}
    diag.setdefault("malformedJsonCells", 0)
    diag.setdefault("cellsScanned", 0)

    def _scan(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        diag["cellsScanned"] = int(diag["cellsScanned"]) + 1
        parsed, malformed = _maybe_json_checked(value)
        if malformed:
            diag["malformedJsonCells"] = int(diag["malformedJsonCells"]) + 1
        return parsed

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    low = {str(c).strip().lower() for c in df.columns}
    if (low & set(_GROUP_COLUMN_NAMES)) and (low & set(_TOOL_COLUMN_NAMES)):
        return None  # already the per-episode trace table
    records: Optional[List[Any]] = None
    if df.shape[1] == 1:
        # A single JSON-ish column of span/observation objects.
        parsed = [_scan(v) for v in df[df.columns[0]].tolist()]
        n_dicts = sum(1 for p in parsed if isinstance(p, dict))
        if n_dicts >= max(1, int(0.8 * len(parsed))):
            records = parsed
    if records is None:
        has_ids = bool(low & set(_TRACE_ID_KEYS))
        has_span_payload = bool(low & set(_PAYLOAD_COLUMNS)) or bool(low & set(_SPAN_ID_COLUMNS))
        span_shaped = (
            (has_ids and (has_span_payload or "name" in low))
            or "observations" in low
            or "attributes" in low
        )
        if not span_shaped:
            return None
        records = []
        for row in df.to_dict(orient="records"):
            rec: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                v = _scan(v)
                rec[str(k)] = v
            records.append(rec)
    flat = parse_trace_export(records)
    if flat is None or len(flat) == 0:
        return None
    steps_total = float(pd.to_numeric(flat["steps"], errors="coerce").fillna(0).sum())
    if steps_total <= 0:
        return None  # nothing event-like was found: not a span export
    return flat


def ingest_and_run_pulse(df: Any, inputs: Dict[str, Any], progress=None) -> Dict[str, Any]:
    """Package-level ``run_pulse``: the G-10 trace-ingestion front door.

    When the run is declared as agent traces (or undeclared) and the
    frame is a raw span/observation export, it is flattened into the
    per-episode table BEFORE the orchestrator's column detection, and
    the ingestion form is disclosed to the agent probe through
    ``inputs["_trace_ingestion"]``. Every other frame passes through
    untouched. The ingestion side never raises; the orchestrator's
    ``run_pulse`` remains the single source of truth for the analysis.
    """
    from vfairness.operations.pulse import orchestrator as _orchestrator

    diagnostics: Dict[str, Any] = {}

    def _ingest():
        src = str(
            (inputs or {}).get("source_kind") or (inputs or {}).get("sourceKind") or ""
        ).lower()
        if src not in _AGENT_SOURCE_KINDS:
            return None
        flat = maybe_flatten_spans(df, diagnostics)
        if flat is None:
            return None
        meta = dict(flat.attrs.get("ingestion") or {})
        new_inputs = dict(inputs or {})
        new_inputs["_trace_ingestion"] = {
            "form": str(meta.get("form") or "otel_spans"),
            "notes": str(
                meta.get("notes") or "Flattened a raw span export into the per-episode trace table."
            ),
        }
        return (flat, new_inputs)

    prepared = _safe(_ingest, None)
    if prepared is None:
        result = _orchestrator.run_pulse(df, inputs, progress=progress)
        # The malformed-export disclosure has to reach the REPORT, not only a
        # Python warning nobody reads in a consumer process. Without this the
        # user gets an ordinary tabular report on a truncated trace export and
        # nothing in it says the traces were never read (see
        # MalformedTraceExportWarning for the measurement).
        return _attach_ingestion_disclosure(result, diagnostics)
    flat_df, new_inputs = prepared
    result = _orchestrator.run_pulse(flat_df, new_inputs, progress=progress)
    return _attach_ingestion_disclosure(result, diagnostics)


def _attach_ingestion_disclosure(result: Any, diagnostics: Dict[str, Any]) -> Any:
    """Record a malformed-JSON count in the pulse result. Never raises."""

    def _attach():
        if not int(diagnostics.get("malformedJsonCells") or 0):
            return result
        if not isinstance(result, dict):
            return result
        data = result.get("data")
        if not isinstance(data, dict):
            return result
        data["traceIngestion"] = {
            "malformedJsonCells": int(diagnostics.get("malformedJsonCells") or 0),
            "cellsScanned": int(diagnostics.get("cellsScanned") or 0),
            "note": str(diagnostics.get("note") or ""),
        }
        degradations = data.get("degradations")
        if not isinstance(degradations, list):
            degradations = []
            data["degradations"] = degradations
        degradations.append(
            {"stage": "trace_ingestion", "error": str(diagnostics.get("note") or "")[:300]}
        )
        return result

    return _safe(_attach, result)
