"""Contract tests for Pulse standard trace ingestion (G-10, with the
G-43 memory-event fields reserved in v1), multi-agent depth (G-21
remainder: EmergentBiasDetector amplification + GroupthinkDetector
gating) and temporal dynamics (G-37: TemporalTracker).

Deterministic throughout: fixtures are exact counts or seeded (seed 7),
and every detector's own resampling is seeded internally. End-to-end
paths route through run_pulse with source_kind='agent' so the ingestion
front door and the router integration are covered.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.agent_probe import agent_probe_pulse
from vfairness.operations.pulse.traces import (
    EPISODE_COLUMNS,
    maybe_flatten_spans,
    parse_trace_export,
)

SEED = 7


def _pulse(df):
    return run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})["data"]


# ── fixtures ────────────────────────────────────────────────────────────────


def _plan(n_per_group=30):
    """Planted tool skew: A picks 'search' 24/30, B picks 'escalate'
    24/30. Cramér's V on the 2x2 table is 0.6."""
    plan = []
    for i in range(n_per_group):
        plan.append(("A", "search" if i < 24 else "escalate"))
    for i in range(n_per_group):
        plan.append(("B", "escalate" if i < 24 else "search"))
    return plan


def _otel_records(n_per_group=30, memory=True):
    """OTLP-JSON-ish GenAI spans: per trace an invoke_agent span carrying
    the group attribute, an execute_tool span, plus (deterministically)
    a memory-read span every third trace and a delegation span for every
    fifth group-A trace."""
    records = []
    t0 = 1720000000
    for i, (g, tool) in enumerate(_plan(n_per_group)):
        tid = "trace-%04d" % i
        base_ns = int((t0 + 10 * i) * 1e9)
        records.append(
            {
                "trace_id": tid,
                "span_id": "s%da" % i,
                "name": "invoke_agent triage",
                "start_time_unix_nano": base_ns,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.id": "triage",
                    "group": g,
                },
            }
        )
        records.append(
            {
                "trace_id": tid,
                "span_id": "s%db" % i,
                "name": "execute_tool %s" % tool,
                "start_time_unix_nano": base_ns + int(1e9),
                "attributes": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool},
                "status": {"code": "STATUS_CODE_OK"},
            }
        )
        if memory and i % 3 == 0:
            records.append(
                {
                    "trace_id": tid,
                    "span_id": "s%dc" % i,
                    "name": "memory.read",
                    "start_time_unix_nano": base_ns + int(2e9),
                    "attributes": {"gen_ai.operation.name": "memory.read"},
                }
            )
        if g == "A" and i % 5 == 0:
            records.append(
                {
                    "trace_id": tid,
                    "span_id": "s%dd" % i,
                    "name": "handoff",
                    "start_time_unix_nano": base_ns + int(3e9),
                    "attributes": {"gen_ai.agent.delegate_to": "senior_agent"},
                }
            )
    return records


def _langfuse_records(n_per_group=30):
    """Langfuse-style export: trace objects with group in the trace
    metadata and nested observations; the tool is a TOOL observation."""
    records = []
    for i, (g, tool) in enumerate(_plan(n_per_group)):
        tid = "lf-%04d" % i
        ts = "2026-07-01T09:%02d:%02d.000Z" % (i // 60, i % 60)
        records.append(
            {
                "id": tid,
                "name": "support-flow",
                "metadata": {"group": g},
                "observations": [
                    {
                        "id": "o%da" % i,
                        "traceId": tid,
                        "type": "SPAN",
                        "name": "plan",
                        "startTime": ts,
                    },
                    {
                        "id": "o%db" % i,
                        "traceId": tid,
                        "type": "TOOL",
                        "name": tool,
                        "startTime": ts,
                    },
                    {
                        "id": "o%dc" % i,
                        "traceId": tid,
                        "type": "GENERATION",
                        "name": "answer",
                        "level": "DEFAULT",
                    },
                ],
            }
        )
    return records


def _amplified_df(n_per_group=40):
    """Per-episode traces where each component agent carries a tiny
    output gap between groups but the SYSTEM output gap is 0.6: planted
    emergent amplification."""
    rng = np.random.default_rng(SEED)
    n = 2 * n_per_group
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    is_a = (group == "A").astype(float)
    return pd.DataFrame(
        {
            "group": group,
            "tool": ["respond"] * n,
            "agent_screener_score": 0.50 + 0.02 * is_a + rng.normal(0.0, 0.01, n),
            "agent_writer_score": 0.50 + 0.01 * is_a + rng.normal(0.0, 0.01, n),
            "system_score": 0.20 + 0.60 * is_a + rng.normal(0.0, 0.02, n),
        }
    )


def _drift_df():
    """Planted disparity growth: 10 ordered windows of 15 A + 15 B rows.
    A's 'escalate' selection count rises linearly across windows while
    B stays flat, so the per-window rate gap grows monotonically."""
    rows = []
    t = 0
    for w in range(10):
        esc_a = int(round(15.0 * w / 9.0))
        for j in range(15):
            rows.append(
                {"group": "A", "tool": "escalate" if j < esc_a else "search", "timestamp": t}
            )
            t += 1
        for j in range(15):
            rows.append({"group": "B", "tool": "escalate" if j < 1 else "search", "timestamp": t})
            t += 1
    return pd.DataFrame(rows)


def _no_extra_columns_df():
    return pd.DataFrame({"group": ["A"] * 30 + ["B"] * 30, "tool": ["search", "escalate"] * 30})


# ── G-10: OTel span parsing ─────────────────────────────────────────────────


def test_otel_records_parse_to_per_episode_table():
    flat = parse_trace_export(_otel_records())
    assert list(flat.columns) == list(EPISODE_COLUMNS)
    assert len(flat) == 60
    assert flat.attrs["ingestion"]["form"] == "otel_spans"
    # Group comes from the span attributes; the tool is the FIRST tool
    # invocation per trace and the tool spans are counted.
    r0 = flat.iloc[0]
    assert r0["group"] == "A"
    assert r0["tool"] == "search"
    assert (flat["tool_calls"] == 1.0).all()
    # Trace 0 has agent + tool + memory + delegation spans.
    assert r0["steps"] == 4.0
    # Planted skew survives flattening.
    a = flat[flat["group"] == "A"]
    b = flat[flat["group"] == "B"]
    assert float((a["tool"] == "search").mean()) == pytest.approx(0.8)
    assert float((b["tool"] == "escalate").mean()) == pytest.approx(0.8)
    # Route only where planted; never fabricated elsewhere.
    assert set(flat["route"].dropna()) == {"senior_agent"}
    assert flat["route"].isna().sum() == 60 - 6
    # Outcome derives from the span status; timestamps are the earliest
    # span start per trace and preserve the planted ordering.
    assert (flat["outcome"] == "success").all()
    ts = flat["timestamp"].to_numpy(dtype=float)
    assert np.all(np.diff(ts) > 0)


def test_otlp_keyed_list_attributes_and_container_unwrap():
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "s1",
                                "name": "execute_tool search",
                                "attributes": [
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "search"}},
                                    {"key": "group", "value": {"stringValue": "A"}},
                                ],
                                "startTimeUnixNano": "1720000000000000000",
                            }
                        ]
                    }
                ]
            }
        ]
    }
    flat = parse_trace_export(payload)
    assert len(flat) == 1
    assert flat.iloc[0]["tool"] == "search"
    assert flat.iloc[0]["group"] == "A"
    assert float(flat.iloc[0]["timestamp"]) == pytest.approx(1720000000.0)


def test_otel_spans_route_end_to_end_through_run_pulse():
    d = _pulse(pd.DataFrame(_otel_records()))
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["ingestion"]["form"] == "otel_spans"
    assert "notes" in d["agent"]["ingestion"]
    assert d["agent"]["available"] is True
    assert d["agent"]["sampleAdequacy"]["adequate"] is True
    # The planted skew fires through the standard omnibus + BH gate.
    tool_findings = [f for f in d["bias"] if f["type"] == "agent_tool_bias"]
    assert tool_findings


# ── G-10: Langfuse export parsing ───────────────────────────────────────────


def test_langfuse_records_parse_to_per_episode_table():
    flat = parse_trace_export(_langfuse_records())
    assert flat.attrs["ingestion"]["form"] == "langfuse"
    assert len(flat) == 60
    assert flat.iloc[0]["group"] == "A"
    assert flat.iloc[0]["tool"] == "search"
    assert flat.iloc[0]["steps"] == 3.0
    # No memory-ish observations anywhere: reserved fields stay NaN.
    assert flat["memory_reads"].isna().all()
    assert flat["memory_writes"].isna().all()


def test_langfuse_flat_observation_rows_parse():
    rows = []
    for i, (g, tool) in enumerate(
        [("A", "search"), ("A", "search"), ("B", "escalate"), ("B", "escalate")]
    ):
        tid = "lf-flat-%d" % i
        rows.append(
            {
                "id": "o%da" % i,
                "traceId": tid,
                "type": "SPAN",
                "name": "plan",
                "metadata": {"group": g},
                "startTime": "2026-07-01T10:00:%02d.000Z" % i,
            }
        )
        rows.append(
            {
                "id": "o%db" % i,
                "traceId": tid,
                "type": "TOOL",
                "name": tool,
                "startTime": "2026-07-01T10:00:%02d.500Z" % i,
            }
        )
    flat = parse_trace_export(rows)
    assert len(flat) == 4
    assert flat["group"].tolist() == ["A", "A", "B", "B"]
    assert flat["tool"].tolist() == ["search", "search", "escalate", "escalate"]
    assert (flat["steps"] == 2.0).all()


def test_langfuse_spans_route_end_to_end_through_run_pulse():
    d = _pulse(pd.DataFrame(_langfuse_records()))
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["ingestion"]["form"] == "langfuse"
    assert d["agent"]["available"] is True
    tool_findings = [f for f in d["bias"] if f["type"] == "agent_tool_bias"]
    assert tool_findings


# ── G-43 groundwork: reserved memory-event fields ───────────────────────────


def test_memory_fields_parsed_when_present_else_nan():
    with_mem = parse_trace_export(_otel_records(memory=True))
    no_mem = parse_trace_export(_otel_records(memory=False))
    # Present: counted per trace, 0 (not NaN) for traces without them.
    assert with_mem["memory_reads"].notna().all()
    assert float(with_mem.iloc[0]["memory_reads"]) == 1.0
    assert float(with_mem.iloc[1]["memory_reads"]) == 0.0
    assert with_mem["memory_writes"].notna().all()
    assert float(with_mem["memory_writes"].sum()) == 0.0
    # Absent everywhere: honest NaN, but the columns still exist (the
    # v1 contract reserves them so no v2 contract is needed).
    assert "memory_reads" in no_mem.columns
    assert "memory_writes" in no_mem.columns
    assert no_mem["memory_reads"].isna().all()
    assert no_mem["memory_writes"].isna().all()


def test_memory_write_verbs_counted_as_writes():
    records = [
        {"trace_id": "t1", "span_id": "a", "name": "agent", "attributes": {"group": "A"}},
        {
            "trace_id": "t1",
            "span_id": "b",
            "name": "memory.store",
            "attributes": {"gen_ai.operation.name": "memory.store"},
        },
        {"trace_id": "t1", "span_id": "c", "name": "retrieve_memory", "attributes": {}},
    ]
    flat = parse_trace_export(records)
    assert float(flat.iloc[0]["memory_writes"]) == 1.0
    assert float(flat.iloc[0]["memory_reads"]) == 1.0


# ── G-10: per-episode passthrough and disclosure ────────────────────────────


def test_per_episode_frame_is_never_flattened_and_disclosed():
    df = _no_extra_columns_df()
    assert maybe_flatten_spans(df) is None
    d = _pulse(df)
    assert d["agent"]["ingestion"]["form"] == "per_episode"


# ── G-21: emergent amplification ────────────────────────────────────────────


def test_amplification_fires_on_system_amplified_traces():
    d = _pulse(_amplified_df())
    amp = d["agent"]["amplification"]
    assert amp["available"] is True
    assert amp["detector"] == "EmergentBiasDetector"
    assert amp["systemColumn"] == "system_score"
    assert set(amp["componentColumns"]) == {"agent_screener_score", "agent_writer_score"}
    comp = amp["perComparison"][0]
    assert comp["isEmergent"] is True
    assert comp["isSignificant"] is True
    assert comp["amplificationFactor"] > 1.5
    assert comp["systemBias"] > comp["maxComponentBias"]
    findings = [f for f in d["bias"] if f["type"] == "emergent_amplification"]
    assert findings
    # A ~20x amplification is critical by the factor-derived severity.
    assert findings[0]["severity"] == "critical"


def test_amplification_not_applicable_without_columns():
    d = _pulse(_no_extra_columns_df())
    amp = d["agent"]["amplification"]
    assert amp["available"] is False
    assert amp["notApplicable"] is True
    assert "per-agent AND system outputs" in amp["reason"]
    assert [f for f in d["bias"] if f["type"] == "emergent_amplification"] == []


# ── G-21: groupthink honesty gate ───────────────────────────────────────────


def test_groupthink_not_applicable_on_flat_unordered_rows():
    # Two agent-opinion columns exist but there is no ordering column,
    # so the GroupthinkDetector's per-round API cannot honestly run.
    d = _pulse(_amplified_df())
    gt = d["agent"]["groupthink"]
    assert gt["available"] is False
    assert gt["notApplicable"] is True
    assert "ordering" in gt["reason"]
    assert [f for f in d["bias"] if f["type"] == "groupthink_convergence"] == []


def test_groupthink_not_applicable_without_opinion_columns():
    d = _pulse(_no_extra_columns_df())
    gt = d["agent"]["groupthink"]
    assert gt["available"] is False
    assert gt["notApplicable"] is True
    assert "opinion columns" in gt["reason"]


# ── G-37: temporal dynamics ─────────────────────────────────────────────────


def test_temporal_drift_fires_with_time_column():
    d = _pulse(_drift_df())
    td = d["agent"]["temporalDynamics"]
    assert td["available"] is True
    assert td["timeColumn"] == "timestamp"
    assert td["driftDetected"] is True
    assert "TemporalTracker" in td["method"]
    assert td["windows"] >= 3
    assert len(td["perWindow"]) == td["windows"]
    assert set(td["groups"]) == {"A", "B"}
    # The disparity series grows monotonically by construction.
    disparities = [w["disparity"] for w in td["perWindow"]]
    assert disparities[-1] > disparities[0]
    assert td["feedbackLoop"]["hasFeedbackLoop"] is True
    assert td["feedbackLoop"]["trendDirection"] == "increasing"
    findings = [f for f in d["bias"] if f["type"] == "agent_temporal_drift"]
    assert findings
    # The final-window gap is ~0.93: critical by the gap convention.
    assert findings[0]["severity"] == "critical"
    assert findings[0]["statisticalTest"]["detector"] == "TemporalTracker"


def test_temporal_not_applicable_without_time_column():
    d = _pulse(_no_extra_columns_df())
    td = d["agent"]["temporalDynamics"]
    assert td["available"] is False
    assert td["notApplicable"] is True
    assert "timestamp/ordering column" in td["reason"]
    assert [f for f in d["bias"] if f["type"] == "agent_temporal_drift"] == []


# ── malformed exports degrade honestly ──────────────────────────────────────


def test_malformed_span_records_degrade_honestly_through_run_pulse():
    df = pd.DataFrame(
        {
            "trace_id": [None, None, None],
            "span_id": ["a", "b", "c"],
            "attributes": ["{broken json", 7, None],
        }
    )
    r = run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})
    assert r["success"] is True
    d = r["data"]
    assert d["sourceKind"] == "agent_traces"
    assert d["verdict"]["tone"] == "disclaimer"
    assert d["bias"] == []


def test_partially_malformed_records_are_skipped_not_fatal():
    records = _otel_records(n_per_group=30)
    records.insert(0, "not a dict")
    records.insert(1, {"span_id": "orphan", "name": "no trace id"})
    flat = parse_trace_export(records)
    assert len(flat) == 60
    assert "skipped" in flat.attrs["ingestion"]["notes"]


def test_probe_direct_call_on_unusable_frame_returns_valid_payload():
    out = agent_probe_pulse(pd.DataFrame({"x": [1, 2, 3]}), {}, "tool", "group", "", "")
    assert out["success"] is True
    agent = out["data"]["agent"]
    assert agent["available"] is False
    assert "ingestion" in agent
    assert agent["amplification"]["available"] is False
    assert agent["temporalDynamics"]["available"] is False
    assert out["data"]["assurance"]["overall"] == "Disclaimer"
