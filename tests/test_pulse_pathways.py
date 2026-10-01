"""Pathway contract + independent-oracle tests for the Pulse orchestrator.

Covers the pathways test_pulse_orchestrator.py does not: agent traces
(per-episode table, raw OpenTelemetry GenAI spans, Langfuse exports) and image
manifests, plus the tier-stable half of the generative screen.

Two things make these tests worth trusting:

1. INDEPENDENT ORACLE. Where a metric is deterministic the expected value is
   recomputed here from the raw data and the metric's mathematical definition,
   never from vfairness internals (image MaxSkew == ln(observed/reference);
   the agent's reported tool rates == raw group counts). A metric bug therefore
   cannot hide behind the engine checking itself.

2. TIER STABILITY. The package's declared dependencies (numpy / pandas / scipy /
   requests / scikit-learn) do NOT include vaderSentiment, detoxify or
   transformers, so this suite runs the LEXICAL floor. Assertions lean on
   tier-stable signals: the pattern-based RefusalScorer, deterministic skew
   math, and raw tool counts -- not sentiment magnitudes, which move with the
   installed scorer tier.

All data is built inline and deterministically (no RNG), so a failure is a real
regression, never a reseed.
"""

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from vfairness.operations.pulse import run_pulse

# ── Agent traces ───────────────────────────────────────────────────────────
# Planted: men are auto-approved and escalated far more; women are pushed to
# more documentation, denials and collections. 100 episodes per group so the
# omnibus chi-square + BH correction have power.

_TOOLS_MAN = (
    ["auto_approve"] * 45
    + ["escalate_to_specialist"] * 30
    + ["request_more_docs"] * 15
    + ["auto_deny"] * 10
)
_TOOLS_WOMAN = (
    ["auto_approve"] * 20
    + ["escalate_to_specialist"] * 13
    + ["request_more_docs"] * 30
    + ["auto_deny"] * 25
    + ["route_to_collections"] * 12
)
_BASE_TS = 1_780_000_000


def _agent_episodes():
    rows = []
    for i, tool in enumerate(_TOOLS_MAN):
        rows.append(
            {
                "trace_id": f"m{i:03d}",
                "gender": "man",
                "tool": tool,
                "route": "tier2_human" if i % 10 < 7 else "tier1_bot",
                "steps": 6 + (i % 4),
                "outcome": "success" if i % 10 < 8 else "failure",
                "timestamp": _BASE_TS + i * 60,
            }
        )
    for i, tool in enumerate(_TOOLS_WOMAN):
        rows.append(
            {
                "trace_id": f"w{i:03d}",
                "gender": "woman",
                "tool": tool,
                "route": "tier1_bot" if i % 10 < 6 else "collections",
                "steps": 3 + (i % 4),
                "outcome": "success" if i % 10 < 5 else "failure",
                "timestamp": _BASE_TS + 10_000 + i * 60,
            }
        )
    return rows


def _agent_df() -> pd.DataFrame:
    return pd.DataFrame(_agent_episodes())


def _otel_spans():
    """The same episodes as a raw OpenTelemetry GenAI span export."""
    spans = []
    for e in _agent_episodes():
        t0 = e["timestamp"] * 1_000_000_000
        spans.append(
            {
                "trace_id": e["trace_id"],
                "span_id": f"{e['trace_id']}-s0",
                "name": "invoke_agent triage",
                "start_time_unix_nano": t0,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.id": "triage",
                    "group": e["gender"],
                },
                "status": {
                    "code": "STATUS_CODE_ERROR" if e["outcome"] == "failure" else "STATUS_CODE_OK"
                },
            }
        )
        spans.append(
            {
                "trace_id": e["trace_id"],
                "span_id": f"{e['trace_id']}-s1",
                "name": f"execute_tool {e['tool']}",
                "start_time_unix_nano": t0 + 1_000_000_000,
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": e["tool"],
                },
            }
        )
    return spans


def _langfuse_traces():
    """The same episodes as a Langfuse trace/observation export."""
    out = []
    for e in _agent_episodes():
        base = datetime.fromtimestamp(e["timestamp"], tz=timezone.utc)

        def iso(offset, base=base):
            return (base + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")

        out.append(
            {
                "id": e["trace_id"],
                "name": "benefits-triage",
                "metadata": {"group": e["gender"]},
                "observations": [
                    {
                        "id": f"{e['trace_id']}-o0",
                        "traceId": e["trace_id"],
                        "type": "SPAN",
                        "name": "plan",
                        "startTime": iso(0),
                    },
                    {
                        "id": f"{e['trace_id']}-o1",
                        "traceId": e["trace_id"],
                        "type": "TOOL",
                        "name": e["tool"],
                        "startTime": iso(1),
                        "level": "ERROR" if e["outcome"] == "failure" else "DEFAULT",
                    },
                ],
            }
        )
    return out


def _agent_inputs(attr: str):
    return {
        "system_type": "agentic",
        "source_kind": "agent",
        "protected_attributes": [attr],
        "domain": "Financial services",
        "jurisdiction": "EU",
    }


def _finding_types(result) -> set:
    return {
        str(f.get("type") or f.get("name")) for f in result["data"]["assurance"].get("findings", [])
    }


def test_agent_per_episode_flags_tool_and_action_disparity():
    d = run_pulse(_agent_df(), _agent_inputs("gender"))["data"]
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["available"] is True
    assert d["agent"]["ingestion"]["form"] == "per_episode"
    types = {str(f.get("type") or f.get("name")) for f in d["assurance"]["findings"]}
    assert any("tool" in t for t in types), types
    assert any("action" in t for t in types), types
    # a planted, family-wise-significant disparity must not read as Clean
    assert d["assurance"]["overall"] != "Clean"


def test_agent_otel_spans_are_flattened_then_analysed():
    d = run_pulse(pd.DataFrame(_otel_spans()), _agent_inputs("group"))["data"]
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["ingestion"]["form"] == "otel_spans"
    # 2 spans per episode collapse back to one row per trace
    assert d["agent"]["sampleAdequacy"]["perGroup"] == {"man": 100, "woman": 100}
    types = {str(f.get("type") or f.get("name")) for f in d["assurance"]["findings"]}
    assert any("tool" in t for t in types), types


def test_agent_langfuse_export_is_flattened_then_analysed():
    d = run_pulse(pd.DataFrame(_langfuse_traces()), _agent_inputs("group"))["data"]
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["ingestion"]["form"] == "langfuse"
    assert d["agent"]["sampleAdequacy"]["perGroup"] == {"man": 100, "woman": 100}
    types = {str(f.get("type") or f.get("name")) for f in d["assurance"]["findings"]}
    assert any("tool" in t for t in types), types


def test_agent_reported_tool_rates_match_raw_counts():
    """Independent oracle: the engine's reported auto_approve rates must equal
    the rates recomputed straight from the raw episode table."""
    import re

    df = _agent_df()
    rate_man = round((df[df.gender == "man"].tool == "auto_approve").mean() * 100)
    rate_woman = round((df[df.gender == "woman"].tool == "auto_approve").mean() * 100)
    assert (rate_man, rate_woman) == (45, 20)  # the planted design, recomputed

    d = run_pulse(df, _agent_inputs("gender"))["data"]
    evidence = next(
        (f.get("evidence") or f.get("plain") or "")
        for f in d["assurance"]["findings"]
        if "tool" in str(f.get("type"))
    )
    pcts = [int(x) for x in re.findall(r"(\d+)%", evidence)]
    assert rate_man in pcts, (rate_man, pcts, evidence)
    assert rate_woman in pcts, (rate_woman, pcts, evidence)


# ── Image manifests ────────────────────────────────────────────────────────


def _image_df(n_man: int = 30, n_woman: int = 10) -> pd.DataFrame:
    rows = [
        {
            "image_id": f"m{i:03d}",
            "image_url": f"https://example.invalid/m{i}.jpg",
            "detected_gender": "man",
        }
        for i in range(n_man)
    ]
    rows += [
        {
            "image_id": f"w{i:03d}",
            "image_url": f"https://example.invalid/w{i}.jpg",
            "detected_gender": "woman",
        }
        for i in range(n_woman)
    ]
    return pd.DataFrame(rows)


_IMAGE_INPUTS = {
    "system_type": "generative",
    "source_kind": "image",
    "ingestion_mode": "image_manifest",
    "domain": "Marketing imagery",
    "jurisdiction": "EU",
}


def test_image_manifest_flags_representation_skew():
    d = run_pulse(_image_df(), _IMAGE_INPUTS)["data"]
    assert d["sourceKind"] == "image_set"
    assert d["vision"]["available"] is True
    types = {str(f.get("type") or f.get("name")) for f in d["assurance"]["findings"]}
    assert any("skew" in t for t in types), types


def test_image_skew_equals_independent_log_ratio_recomputation():
    """Independent oracle: MaxSkew is ln(observed / reference) (Geyik 2019).
    Recompute it from the raw labels and a uniform reference, and require the
    engine to agree. A pre-labeled manifest is never fetched, so the URLs here
    are unreachable on purpose."""
    df = _image_df(n_man=30, n_woman=10)
    labels = df["detected_gender"].tolist()
    n = len(labels)
    p_man = labels.count("man") / n  # 0.75
    p_woman = labels.count("woman") / n  # 0.25
    reference = 0.5  # 2 groups, uniform
    expected_max_skew = math.log(p_man / reference)  # ln(1.5) == 0.4055

    skew = run_pulse(df, _IMAGE_INPUTS)["data"]["vision"]["skew"]
    assert skew["available"] is True
    assert abs(skew["observed"]["man"] - p_man) < 1e-9
    assert abs(skew["observed"]["woman"] - p_woman) < 1e-9
    assert abs(skew["maxSkew"] - expected_max_skew) < 1e-3, (skew["maxSkew"], expected_max_skew)
    assert abs(skew["perGroup"]["woman"] - math.log(p_woman / reference)) < 1e-3
    assert skew["mostOverrepresented"] == "man"
    assert skew["mostUnderrepresented"] == "woman"


def test_image_balanced_manifest_is_not_skewed():
    """Metamorphic counterpart: an even split must recompute to zero skew."""
    skew = run_pulse(_image_df(n_man=20, n_woman=20), _IMAGE_INPUTS)["data"]["vision"]["skew"]
    assert abs(skew["maxSkew"] - math.log(0.5 / 0.5)) < 1e-9  # ln(1) == 0


# ── Generative: the tier-stable refusal screen ─────────────────────────────
# RefusalScorer is pattern-based at every tier, so a planted refusal gap is a
# stable assertion; sentiment magnitude is not and is deliberately not asserted.

_REFUSAL = (
    "I'm sorry, but I'm not able to help with that request. "
    "You might want to consult our help center instead."
)
_WARM = (
    "Absolutely, I'd be happy to help! Great question. I've taken care of it "
    "and everything looks good now. Let me know if there is anything else."
)


def _generative_refusal_df(n: int = 120, woman_refusal_rate: float = 0.6) -> pd.DataFrame:
    rows = [
        {"prompt": "How do I reset my password?", "response": _WARM, "gender": "man"}
        for _ in range(n)
    ]
    cutoff = int(woman_refusal_rate * n)
    rows += [
        {
            "prompt": "How do I reset my password?",
            "response": _REFUSAL if i < cutoff else _WARM,
            "gender": "woman",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


_GEN_INPUTS = {
    "system_type": "generative",
    "source_kind": "generative",
    "ingestion_mode": "prompts_outputs",
    "protected_attributes": ["gender"],
    "domain": "Customer support",
    "jurisdiction": "EU",
}


def test_generative_refusal_disparity_is_detected():
    df = _generative_refusal_df()
    # independent recomputation of the planted gap (no vfairness scorers)
    hints = ("i'm not able", "i cannot", "i can't", "i'm sorry, but")
    refuses = df.response.map(lambda t: any(h in t.lower() for h in hints))
    r_woman = refuses[df.gender == "woman"].mean()
    r_man = refuses[df.gender == "man"].mean()
    assert (round(r_woman, 2), r_man) == (0.6, 0.0)

    d = run_pulse(df, _GEN_INPUTS)["data"]
    assert d["sourceKind"] == "prompts_outputs"
    assert d["generative"]["available"] is True
    # a 60-vs-0 refusal gap must surface, not read as "no disparity"
    assert "no significant" not in (d["generative"].get("summary") or "").lower()
    assert d["assurance"]["overall"] != "Clean"


def test_generative_equal_treatment_does_not_invent_a_refusal_gap():
    """Metamorphic counterpart: identical responses per group must not fire."""
    df = _generative_refusal_df(woman_refusal_rate=0.0)
    d = run_pulse(df, _GEN_INPUTS)["data"]
    assert d["generative"]["available"] is True
    types = {str(f.get("type") or f.get("name")) for f in d["assurance"]["findings"]}
    assert not any("refusal" in t.lower() for t in types), types
