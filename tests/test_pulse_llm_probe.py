"""Contract tests for the Pulse LLM-endpoint probe (G-07) and the
benchmark provenance fields (G-30).

Every test runs against a local stdlib http.server stub (same pattern as
tests/test_pulse_orchestrator.py): deterministic, no network, no external services.
The stub speaks the OpenAI chat-completions shape and answers with
strongly polarised plain-keyword sentences so the library's built-in
keyword sentiment scorer separates the arms cleanly.
"""

import http.server
import json
import threading

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
    return out["data"]


# ── (a) + (b): multi-template protocol, replication, family-wise stats ────


def test_multi_template_replication_and_family_wise_significance():
    """Stub answers negatively for Jamal/Lakisha AND the older-age
    descriptor across EVERY template: the name and age axes must be
    family-wise significant and replicated in >= 2 templates, while the
    religion and disability axes stay quiet."""

    def decide(body):
        neg = "Jamal" in body or "Lakisha" in body or "a 61-year-old" in body
        return _NEG if neg else _POS

    d = _probe_against(decide)
    assert d["sourceKind"] == "llm_endpoint"
    gen = d["generative"]
    assert gen["available"] is True

    # (a) multiple templates ran, with a reproducibility protocol block.
    proto = gen["protocol"]
    assert proto["templates"] >= 3
    assert proto["runsPerArm"] == 2
    assert proto["maxCalls"] == 250
    assert proto["estimatedCalls"] <= proto["maxCalls"]
    assert proto["model_name"] == "stub"
    assert set(proto["axes"]) >= {"name", "age", "disability", "religion"}

    rows = {r["axis"]: r for r in gen["perAxis"]}
    for row in rows.values():
        for key in ("templatesTested", "replicatedIn", "medianEffectSize", "familyWiseSignificant"):
            assert key in row, "perAxis row missing {0}".format(key)
        assert row["templatesTested"] >= 3

    # (b) name + age fire family-wise and replicate; religion/disability
    # do not.
    for axis in ("name", "age"):
        assert rows[axis]["familyWiseSignificant"] is True, axis
        assert rows[axis]["replicatedIn"] >= 2, axis
    for axis in ("religion", "disability"):
        assert rows[axis]["familyWiseSignificant"] is False, axis
        assert rows[axis]["replicatedIn"] == 0, axis

    # Findings carry computed severity: fully replicated with maximal
    # effect size must be "critical" (this fails if severity were still
    # hardcoded to "high").
    sev = {f["attribute"]: f["severity"] for f in d["bias"]}
    assert set(sev) == {"name", "age"}
    assert sev["name"] == "critical"
    assert sev["age"] == "critical"
    assert d["verdict"]["tone"] in ("critical", "warn")
    assert d["scope"]["pathway"] == "llm_endpoint"
    assert "2 runs per arm" in d["scope"]["powerNote"]


# ── (c): one-template signal is capped at warn, marked not replicated ─────


def test_single_template_signal_is_capped_at_warn():
    """Stub answers negatively for Jamal in exactly ONE template (the
    shortlist decision). The name axis fires family-wise but replicates in
    only 1 template, so severity is capped at "warn" with the honest
    not-replicated wording, never a hardcoded "high"."""

    def decide(body):
        neg = ("Jamal" in body) and ("makes the shortlist" in body)
        return _NEG if neg else _POS

    d = _probe_against(decide)
    rows = {r["axis"]: r for r in d["generative"]["perAxis"]}
    assert rows["name"]["familyWiseSignificant"] is True
    assert rows["name"]["replicatedIn"] == 1
    assert rows["name"]["templatesTested"] >= 3

    findings = [f for f in d["bias"] if f["attribute"] == "name"]
    assert len(findings) == 1
    assert findings[0]["severity"] == "warn"
    assert findings[0]["severity"] != "high"
    assert "not replicated across templates" in findings[0]["plain"]
    assert findings[0]["replicatedIn"] == 1


# ── (d): empty responses => honest Disclaimer, never a fabricated finding ─


def test_empty_responses_yield_honest_disclaimer():
    """A stub that always answers with empty content (the reasoning-model
    failure mode) must produce a Disclaimer naming the empty-response rate
    and suggesting a higher max_tokens, with zero fabricated findings."""

    def decide(body):
        return ""

    d = _probe_against(decide, max_tokens=64)
    assert d["assurance"]["overall"] == "Disclaimer"
    assert d["verdict"]["tone"] == "disclaimer"
    one_line = d["assurance"]["oneLineVerdict"]
    assert "empty" in one_line.lower()
    assert "%" in one_line
    prose = one_line + d["verdict"]["summary"] + d["nextStep"]
    assert "max_tokens" in prose
    assert d["bias"] == []
    assert d["generative"]["available"] is False
    assert d["generative"]["emptyResponseRate"] > 0.5
    assert d["generative"]["callsMade"] > 0


# ── (e): benchmark provenance fields (G-30) ────────────────────────────────


def test_benchmark_provenance_fields():
    from vfairness.llm.benchmarks import BenchmarkRunner, benchmark_provenance

    # The pure helper is callable without any endpoint.
    prov = benchmark_provenance("bbq", 25)
    assert prov["benchmark"] == "bbq"
    assert prov["subsetSize"] == 25
    assert prov["fullBenchmarkSize"] == 58000
    assert "bundled subset of 25 items" in prov["provenanceNote"]
    assert "do not present as a full benchmark run" in prov["provenanceNote"]
    assert benchmark_provenance("bold", 4)["fullBenchmarkSize"] == 23000
    assert benchmark_provenance("holistic_bias", 9)["fullBenchmarkSize"] == 460000
    dt = benchmark_provenance("decodingtrust", 10)
    assert dt["fullBenchmarkSize"] == "varies by task"
    assert "varies by task" in dt["provenanceNote"]

    # A real result assembly (run_bbq against a trivial local fake proxy)
    # carries the fields end to end, including through to_dict().
    class _FakeProxy:
        def send_prompt(self, prompt, system_prompt=None, temperature=0.0, max_tokens=1024):
            return {"text": "C", "latency_ms": 1.0, "token_count": 1}

    result = BenchmarkRunner(_FakeProxy()).run_bbq(categories=["gender"], sample_size=2)
    rd = result.to_dict()
    assert rd["benchmark"] == "bbq"
    assert rd["subsetSize"] == 2
    assert rd["fullBenchmarkSize"] == 58000
    assert "do not present as a full benchmark run" in rd["provenanceNote"]
