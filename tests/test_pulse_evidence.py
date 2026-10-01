"""Contract tests for Pulse close-plan Phase 4 item G-20: per-finding raw
evidence traces plus the assurance.auditTrail.probe reproducibility
envelope, on both probe pathways.

LLM probe: every test runs against a local stdlib http.server stub (same
pattern as tests/test_pulse_llm_probe.py): deterministic, no network. The
stub speaks the OpenAI chat-completions shape and answers with strongly
polarised plain-keyword sentences so the built-in keyword sentiment
scorer separates the arms cleanly.

Agent probe: exact-count fixtures with planted skews on tool selection,
delegation routing and trajectory steps; the probe's own resampling is
seeded (seed 7), so everything here is deterministic.
"""

import http.server
import json
import threading

import pandas as pd

from vfairness.operations.pulse.agent_probe import agent_probe_pulse
from vfairness.operations.pulse.llm_probe import llm_probe_pulse

# Bare sentiment keywords (no punctuation glued onto them) so the built-in
# KeywordSentimentScorer scores these +1.0 / -1.0 deterministically.
_POS = (
    "The profile is excellent and the fit is good so this is an "
    "outstanding candidate and we are pleased to recommend proceeding"
)
_NEG = (
    "The profile is poor and the fit is bad so this is a terrible "
    "candidate and we do not recommend proceeding"
)
# Long enough to trip the 600-char evidence truncation, still all-negative.
_NEG_LONG = _NEG + " The outlook remains bad and the record is poor." * 20


def _serve(decide):
    """Start a local OpenAI-shaped stub; ``decide(body_text) -> content``."""

    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = (self.rfile.read(n) or b"").decode("utf-8", "replace")
            content = decide(body)
            resp = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _probe_against(decide, **extra_cfg):
    srv = _serve(decide)
    try:
        port = srv.server_address[1]
        cfg = {
            "endpoint_url": "http://127.0.0.1:{0}/v1/chat/completions".format(port),
            "api_format": "openai",
            "model_name": "stub",
            "n_runs": 2,
            # The stub is on loopback; the egress guard refuses 127.0.0.1 unless a
            # caller opts in explicitly. Production callers must NOT set this.
            "allow_loopback": True,
        }
        cfg.update(extra_cfg)
        out = llm_probe_pulse(cfg)
    finally:
        srv.shutdown()
    assert out.get("success") is True
    return out


# == LLM probe: evidence samples + auditTrail.probe =========================


def test_llm_finding_carries_literal_prompt_response_evidence():
    """A stub that answers negatively for Jamal only must produce a name
    finding whose evidence holds the ACTUAL prompts and responses (the
    stub's literal wording), the template id, both arms, and 600-char
    truncation with an ellipsis note on the long negative reply."""

    def decide(body):
        return _NEG_LONG if "Jamal" in body else _POS

    out = _probe_against(decide)
    d = out["data"]
    findings = [f for f in d["bias"] if f["attribute"] == "name"]
    assert len(findings) == 1
    evidence = findings[0]["evidence"]
    assert 1 <= len(evidence) <= 3

    template_ids = set(d["generative"]["protocol"]["templateIds"])
    for sample in evidence:
        # The template id is a real protocol template.
        assert sample["template"] in template_ids
        # The divergent arm is the planted one, against the reference arm.
        assert sample["arm"] == "Jamal"
        assert sample["referenceArm"] == "Greg"
        # The prompt is the literal filled template that went to the
        # endpoint (it names the arm).
        assert "Jamal" in sample["prompt"]
        # The responses are the stub's ACTUAL wording, not paraphrase.
        assert "terrible" in sample["response"]
        assert "do not recommend" in sample["response"]
        assert "outstanding" in sample["referenceResponse"]
        assert "pleased to recommend" in sample["referenceResponse"]
        # The long negative reply is truncated at 600 chars with a note.
        assert sample["response"].endswith("[... truncated at 600 chars]")
        assert len(sample["response"]) <= 600 + len(" [... truncated at 600 chars]")
        # The short positive reply is NOT truncated.
        assert "truncated" not in sample["referenceResponse"]

    # Evidence samples come from distinct template cells, deterministically
    # in template order.
    assert len({s["template"] for s in evidence}) == len(evidence)

    # The FULL result payload stays JSON-serializable.
    json.dumps(out)


def test_llm_audit_trail_probe_envelope():
    """assurance.auditTrail.probe carries the full reproducibility
    envelope, including an honest scorerTier."""

    def decide(body):
        return _NEG if "Jamal" in body else _POS

    d = _probe_against(decide)["data"]
    probe = d["assurance"]["auditTrail"]["probe"]

    assert probe["templates"] >= 3
    assert probe["templates"] == len(probe["templateIds"])
    assert set(probe["axes"]) >= {"name", "age", "disability", "religion"}
    assert probe["runsPerArm"] == 2
    assert probe["maxTokens"] == 1024
    assert probe["modelName"] == "stub"
    assert probe["emptyResponses"] == 0
    assert probe["callsMade"] > 0

    # scorerTier is honest: it must agree with the library's own
    # scorer_status() report, never overstate the scorer in use.
    assert probe["scorerTier"] in ("keyword", "vader", "transformer")
    from vfairness.llm import scorer_status

    desc = str(scorer_status()["sentiment"]["scorer"]).lower()
    if "keyword" in desc:
        assert probe["scorerTier"] == "keyword"
    elif "vader" in desc:
        assert probe["scorerTier"] == "vader"
    else:
        assert probe["scorerTier"] == "transformer"


# == Agent probe: raw trace evidence + auditTrail.probe =====================


def _skewed_agent_df():
    """Two groups, n=60 each, with planted skews on all three surfaces:
    tool selection (search vs escalate), delegation routing (senior vs
    junior) and trajectory step counts (short vs long episodes)."""
    rows = []
    for g, tool_hi, route_hi, base in (
        ("A", "search", "senior_agent", 2.0),
        ("B", "escalate", "junior_agent", 9.0),
    ):
        tool_lo = "escalate" if tool_hi == "search" else "search"
        route_lo = "junior_agent" if route_hi == "senior_agent" else "senior_agent"
        for i in range(60):
            rows.append(
                {
                    "group": g,
                    "tool": tool_hi if i < 50 else tool_lo,
                    "route": route_hi if i < 50 else route_lo,
                    "steps": base + (i % 3),
                }
            )
    return pd.DataFrame(rows)


def test_agent_findings_carry_raw_trace_rows_from_both_groups():
    out = agent_probe_pulse(_skewed_agent_df(), {}, "tool", "group", "", "")
    assert out.get("success") is True
    d = out["data"]
    assert d["agent"]["omnibus"]["significant"] is True

    # Tool finding: first rows per side, real tool values, both groups.
    tool_findings = [f for f in d["bias"] if f["type"] == "agent_tool_bias"]
    assert tool_findings
    for f in tool_findings:
        ev = f["evidence"]
        assert {r["group"] for r in ev} == {"A", "B"}
        assert all(r["tool"] in ("search", "escalate") for r in ev)
        # Up to 5 rows per side, deterministic (first rows, table order):
        # group A's first 5 rows are all 'search'.
        a_rows = [r for r in ev if r["group"] == "A"]
        b_rows = [r for r in ev if r["group"] == "B"]
        assert len(a_rows) == 5 and len(b_rows) == 5
        assert all(r["tool"] == "search" for r in a_rows)
        assert all(r["tool"] == "escalate" for r in b_rows)

    # Action finding shares the same raw rows behind its comparison.
    action_findings = [f for f in d["bias"] if f["type"] == "agent_action_bias"]
    assert action_findings
    assert {r["group"] for r in action_findings[0]["evidence"]} == {"A", "B"}

    # Delegation finding: real route values from both groups.
    deleg_findings = [f for f in d["bias"] if f["type"] == "delegation_routing"]
    assert deleg_findings
    dev = deleg_findings[0]["evidence"]
    assert {r["group"] for r in dev} == {"A", "B"}
    assert all(r["route"] in ("senior_agent", "junior_agent") for r in dev)
    assert [r["route"] for r in dev if r["group"] == "A"] == (["senior_agent"] * 5)

    # Trajectory finding: group + the step value of the same trace row.
    traj_findings = [f for f in d["bias"] if f["type"] == "agent_trajectory_bias"]
    assert traj_findings
    tev = traj_findings[0]["evidence"]
    assert {r["group"] for r in tev} == {"A", "B"}
    assert all(r["column"] == "steps" for r in tev)
    assert all(isinstance(r["value"], float) for r in tev)
    # First rows of group A carry the planted short episodes.
    a_vals = [r["value"] for r in tev if r["group"] == "A"]
    assert a_vals == [2.0, 3.0, 4.0, 2.0, 3.0]

    # The FULL result payload stays JSON-serializable.
    json.dumps(out)


def test_agent_audit_trail_probe_envelope():
    d = agent_probe_pulse(_skewed_agent_df(), {}, "tool", "group", "", "")["data"]
    probe = d["assurance"]["auditTrail"]["probe"]

    assert set(probe["groupsCompared"]) == {"A", "B"}
    assert probe["comparisons"] == 1
    assert probe["familySize"] >= 1
    assert probe["omnibus"]["chi2"] > 0
    assert probe["omnibus"]["p"] < 0.05
    assert probe["sampleFloor"] == 20
    assert probe["seeds"] == {"bootstrap": 7, "permutation": 7}


def test_agent_audit_trail_present_on_low_n_disclaimer():
    """Even the honest low-n Disclaimer carries the probe envelope, so
    the audit trail always records what was (not) run."""
    df = pd.DataFrame({"group": ["A"] * 8 + ["B"] * 8, "tool": ["search", "escalate"] * 8})
    d = agent_probe_pulse(df, {}, "tool", "group", "", "")["data"]
    assert d["assurance"]["overall"] == "Disclaimer"
    probe = d["assurance"]["auditTrail"]["probe"]
    assert probe["groupsCompared"] == []
    assert probe["comparisons"] == 0
    assert probe["familySize"] == 0
    assert probe["sampleFloor"] == 20
    assert probe["seeds"] == {"bootstrap": 7, "permutation": 7}
    json.dumps(d)
