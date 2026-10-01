"""Contract tests for the Pulse orchestrator (run_pulse) and its shim.

Phase-1 truth guarantees (Pulse close plan, platform repo
tasks/pulse-improvements.md): routing per source_kind, degradation
recording that downgrades the verdict, disclaimer semantics (never a
pass tone when nothing was assessed), outcome-polarity normalization,
the worst-attribute Pareto axis, and the quarantine of the legacy
fabricating pipeline. Every test runs locally, deterministic, no
network, no external services.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import handle_pulse_run, run_pulse


def _rng():
    return np.random.default_rng(7)


# ── fixtures ────────────────────────────────────────────────────────────────


def _tabular_df(n: int = 600) -> pd.DataFrame:
    """One protected attribute; group B selected at 0.3 vs A at 0.6."""
    rng = _rng()
    half = n // 2
    group = np.array(["A"] * half + ["B"] * half)
    pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.3, half)])
    return pd.DataFrame({"group": group, "prediction": pred, "feature": rng.normal(size=n)})


def _two_attr_df(n: int = 800) -> pd.DataFrame:
    """small_gap ~0.1 listed FIRST; big_gap ~0.6 listed second."""
    rng = _rng()
    half = n // 2
    small = np.array(["A"] * half + ["B"] * half)
    big = np.tile(np.array(["X"] * (half // 2) + ["Y"] * (half // 2)), 2)
    pred = np.zeros(n, dtype=int)
    for i in range(n):
        p = 0.5 + (0.05 if small[i] == "A" else -0.05) + (0.30 if big[i] == "X" else -0.30)
        pred[i] = rng.binomial(1, min(max(p, 0.02), 0.98))
    score = pred * 0.5 + rng.uniform(0, 0.5, n)
    return pd.DataFrame(
        {
            "small_gap": small,
            "big_gap": big,
            "prediction": pred,
            "score": score,
            "feature": rng.normal(size=n),
        }
    )


def _generative_df(texts_a, texts_b) -> pd.DataFrame:
    rows = [{"prompt": "p", "response": t, "gender": "f"} for t in texts_a]
    rows += [{"prompt": "p", "response": t, "gender": "m"} for t in texts_b]
    return pd.DataFrame(rows)


_EN_POS = (
    "The candidate is excellent and we recommend hiring with strong "
    "confidence in the assessment of this application."
)
_EN_NEG = (
    "The candidate is weak and we do not recommend hiring due to the "
    "poor results in the assessment of this application."
)
_DE_TEXT = (
    "Die Bewertung der Kandidatin fiel insgesamt sehr positiv aus, da sie "
    "ueber umfangreiche Erfahrung verfuegt und ihre Kenntnisse im "
    "Gespraech ueberzeugend darstellen konnte."
)


# ── envelope + routing ──────────────────────────────────────────────────────


def test_tabular_envelope_contract():
    r = run_pulse(_tabular_df(), {"protected_attributes": ["group"]})
    assert r["success"] is True
    d = r["data"]
    for key in (
        "verdict",
        "assurance",
        "perVariable",
        "bias",
        "dataQuality",
        "degradations",
        "proxies",
        "intersectional",
        "recommendedDefinition",
        "interventions",
        "nextStep",
    ):
        assert key in d, f"missing contract key: {key}"
    # Exact tone, not set membership: a 0.6 vs 0.3 selection-rate gap is a
    # stark, BH-significant disparity and must read as critical.
    assert d["verdict"]["tone"] == "critical"
    assert isinstance(d["degradations"], list)


def test_generative_routing_via_source_kind():
    df = _generative_df([_EN_POS] * 20, [_EN_NEG] * 20)
    r = run_pulse(df, {"source_kind": "generative", "protected_attributes": ["gender"]})
    d = r["data"]
    assert d["sourceKind"] == "prompts_outputs"
    assert d["generative"]["available"] is True
    assert d["generative"]["perAttribute"], "groups were analyzable"
    # Exact tone for this crafted input: uniformly positive vs uniformly
    # negative texts on 20 samples per group lands at warn (small n keeps it
    # below critical). It must never read clean.
    assert d["verdict"]["tone"] == "warn"


# ── disclaimer semantics (G-04) ─────────────────────────────────────────────


def test_generative_nothing_assessable_is_disclaimer_not_pass():
    # 3 texts per group is below the n>=5 floor: nothing gets analyzed.
    df = _generative_df([_EN_POS] * 3, [_EN_NEG] * 3)
    r = run_pulse(df, {"source_kind": "generative", "protected_attributes": ["gender"]})
    d = r["data"]
    assert d["assurance"]["overall"] == "Disclaimer"
    assert d["verdict"]["tone"] == "disclaimer"
    assert d["verdict"]["tone"] != "pass"


# ── English-lexicon guard (G-05) ────────────────────────────────────────────


def test_generative_non_english_refuses_instead_of_false_clean():
    texts = [f"{_DE_TEXT} Fall {i}." for i in range(12)]
    df = _generative_df(texts[:6], texts[6:])
    r = run_pulse(df, {"source_kind": "generative", "protected_attributes": ["gender"]})
    d = r["data"]
    assert d["generative"]["available"] is False
    assert "English" in (d["generative"].get("reason") or "")
    assert d["verdict"]["tone"] == "disclaimer"


def test_generative_english_not_over_refused():
    df = _generative_df([_EN_POS] * 20, [_EN_NEG] * 20)
    d = run_pulse(df, {"source_kind": "generative", "protected_attributes": ["gender"]})["data"]
    assert d["generative"]["available"] is True
    assert not d["generative"].get("reason")


# ── degradation recording downgrades the verdict (G-08) ─────────────────────


def test_stage_failure_records_degradation_and_downgrades(monkeypatch):
    import vfairness.preprocessing.bias_detection as bd

    def _boom(*args, **kwargs):
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(bd, "detect_temporal_drift", _boom)
    d = run_pulse(_tabular_df(), {"protected_attributes": ["group"]})["data"]
    stages = [g.get("stage") for g in d["degradations"]]
    assert "temporal_drift" in stages
    # A degraded run must never present an unqualified clean opinion.
    assert d["assurance"]["overall"] != "Unqualified"
    assert d["verdict"]["tone"] != "pass"


def test_mechanism_critical_proxy_failure_is_recorded(monkeypatch):
    import vfairness.operations.pulse.orchestrator as orch

    def _boom(*args, **kwargs):
        raise RuntimeError("proxy battery collapsed")

    monkeypatch.setattr(orch, "_proxies", _boom)
    d = run_pulse(_tabular_df(), {"protected_attributes": ["group"]})["data"]
    stages = [g.get("stage") for g in d["degradations"]]
    assert "proxies" in stages
    assert d["proxies"]["available"] is False
    assert d["verdict"]["tone"] != "pass"


# ── outcome polarity (G-27) ─────────────────────────────────────────────────


def test_polarity_flip_names_the_overflagged_group_worst():
    rng = _rng()
    n = 600
    half = n // 2
    group = np.array(["A"] * half + ["B"] * half)
    # prediction=1 is an ADVERSE event (fraud flag); B is over-flagged.
    flag = np.concatenate([rng.binomial(1, 0.1, half), rng.binomial(1, 0.3, half)])
    df = pd.DataFrame({"group": group, "prediction": flag, "feature": rng.normal(size=n)})
    d_default = run_pulse(df, {"protected_attributes": ["group"]})["data"]
    d_flipped = run_pulse(
        df,
        {
            "protected_attributes": ["group"],
            "outcome_polarity": "positive_unfavorable",
        },
    )["data"]
    assert d_default["verdict"]["ranked"][0]["worstGroup"] == "A"  # inverted
    assert d_flipped["verdict"]["ranked"][0]["worstGroup"] == "B"  # correct
    prep = d_flipped["dataPreparation"]
    assert prep["outcomePolarity"] == "positive_unfavorable"
    assert prep["predictionsInvertedForAnalysis"] is True
    # The score sweep is not polarity-aware yet: it must skip honestly.
    assert d_flipped["pareto"]["available"] is False
    assert "polarity" in d_flipped["pareto"]["reason"]


# ── Pareto axis (G-41) ──────────────────────────────────────────────────────


def test_pareto_runs_on_worst_ranked_attribute_not_first():
    d = run_pulse(_two_attr_df(), {"protected_attributes": ["small_gap", "big_gap"]})["data"]
    ranked = d["verdict"]["ranked"]
    assert ranked[0]["attribute"] == "big_gap"
    assert d["pareto"]["available"] is True
    assert d["pareto"]["attribute"] == "big_gap"


# ── thin shim + quarantine (F-01) ───────────────────────────────────────────


def test_shim_runs_orchestrator_and_forwards_inputs():
    csv = _tabular_df(200).to_csv(index=False)
    env = handle_pulse_run(
        {
            "inputs": {
                "protected_attributes": ["group"],
                "outcome_polarity": "positive_unfavorable",
            },
            "artifact": {"inline_csv": csv, "filename": "t.csv"},
        }
    )
    assert env["success"] is True
    d = env["data"]
    # Envelope is not double-wrapped and inputs reached the orchestrator.
    assert "verdict" in d and "data" not in d
    assert d["dataPreparation"]["predictionsInvertedForAnalysis"] is True


def test_shim_missing_artifact_is_honest_error():
    env = handle_pulse_run({"inputs": {}})
    assert env["success"] is False
    assert "artifact" in env["error"].lower()


def test_legacy_pipeline_is_quarantined():
    from vfairness.operations.pulse.pipeline import (
        PulsePipelineInputs,
        run_pulse_pipeline,
    )

    with pytest.raises(RuntimeError, match="quarantined"):
        run_pulse_pipeline(
            PulsePipelineInputs(
                df=pd.DataFrame({"prediction": [1, 0]}),
                system_type="predictive",
                domain="",
                jurisdiction="",
                artifact_label="",
                protected_attributes=[],
            )
        )


# ── Phase 2: pathway reachability + honest refusals ─────────────────────────


def _trace_df(n: int = 120) -> pd.DataFrame:
    rng = _rng()
    half = n // 2
    group = ["A"] * half + ["B"] * half
    tools = (
        ["search"] * (half - 10) + ["escalate"] * 10 + ["search"] * (half - 30) + ["escalate"] * 30
    )
    return pd.DataFrame({"group": group, "tool": tools, "outcome": rng.binomial(1, 0.5, n)})


def _image_manifest_df(n: int = 90) -> pd.DataFrame:
    labels = ["woman"] * 15 + ["man"] * 75
    return pd.DataFrame(
        {"image_path": [f"img_{i}.png" for i in range(n)], "detected_gender": labels}
    )


def test_agent_routing_via_source_kind():
    d = run_pulse(_trace_df(), {"source_kind": "agent", "protected_attributes": ["group"]})["data"]
    assert d["sourceKind"] == "agent_traces"
    assert d["agent"]["available"] is True
    assert d["agent"]["perComparison"]
    assert d["scope"]["pathway"] == "agent_traces"
    assert d["legalFramework"].get("ratioWarn") is not None


def test_declared_agent_on_tabular_refuses_not_falls_through():
    d = run_pulse(_tabular_df(), {"source_kind": "agent", "protected_attributes": ["group"]})[
        "data"
    ]
    assert d["sourceKind"] == "agent_traces"
    assert d["verdict"]["tone"] == "disclaimer"
    assert "tool/action column" in d["verdict"]["summary"]
    # It must NOT have run the tabular battery.
    assert d["perVariable"] == []


def test_declared_generative_on_tabular_refuses():
    d = run_pulse(_tabular_df(), {"source_kind": "generative", "protected_attributes": ["group"]})[
        "data"
    ]
    assert d["sourceKind"] == "prompts_outputs"
    assert d["verdict"]["tone"] == "disclaimer"
    assert "NOT silently analyzed" in d["verdict"]["summary"]


def test_image_routing_and_honest_ndkl():
    d = run_pulse(_image_manifest_df(), {"source_kind": "image"})["data"]
    assert d["sourceKind"] == "image_set"
    assert d["vision"]["labelsSource"] == "perceived_by_classifier"
    nd = d["vision"]["ndkl"]
    assert isinstance(nd, dict) and "available" in nd
    assert d["scope"]["pathway"] == "image_set"


def test_declared_image_without_labels_refuses():
    d = run_pulse(_tabular_df(), {"source_kind": "image"})["data"]
    assert d["sourceKind"] == "image_set"
    assert d["verdict"]["tone"] == "disclaimer"
    assert "demographic-label column" in d["verdict"]["summary"]


def test_endpoint_only_shim_needs_no_dataset():
    env = handle_pulse_run(
        {
            "inputs": {"source_kind": "endpoint", "llm_config": {}},
        }
    )
    assert env["success"] is True
    d = env["data"]
    assert d["sourceKind"] == "llm_endpoint"
    # No endpoint URL -> explicit disclaimer, never a fabricated verdict.
    assert d["verdict"]["tone"] == "disclaimer"
    assert d["generative"]["available"] is False


@pytest.mark.parametrize(
    "fixture,inputs,kind",
    [
        ("tabular", {"protected_attributes": ["group"]}, "tabular"),
        (
            "gen",
            {"source_kind": "generative", "protected_attributes": ["gender"]},
            "prompts_outputs",
        ),
        ("trace", {"source_kind": "agent", "protected_attributes": ["group"]}, "agent_traces"),
        ("image", {"source_kind": "image"}, "image_set"),
    ],
)
def test_canonical_envelope_on_every_pathway(fixture, inputs, kind):
    df = {
        "tabular": _tabular_df,
        "gen": lambda: _generative_df([_EN_POS] * 20, [_EN_NEG] * 20),
        "trace": _trace_df,
        "image": _image_manifest_df,
    }[fixture]()
    d = run_pulse(df, inputs)["data"]
    assert d["sourceKind"] == kind
    for key in (
        "verdict",
        "assurance",
        "scope",
        "legalFramework",
        "proxies",
        "statistical",
        "intersectional",
        "recommendedDefinition",
        "interventions",
        "nextStep",
    ):
        assert key in d, f"{kind} missing {key}"
    assert d["scope"]["level"] == "triage"
    assert isinstance(d["scope"]["disclosures"], list)
    lf = d["legalFramework"]
    assert set(lf) >= {"framework", "ratioWarn", "ratioCritical", "mode", "domain", "jurisdiction"}


def test_endpoint_probe_full_pipeline_against_local_stub():
    """Drives the WHOLE live-endpoint pathway (B1) against a local
    OpenAI-shaped stub: request building, response extraction, per-axis
    counterfactual comparison, findings, verdict. Deterministic, CI-safe.
    The stub answers curtly negative for two of the probe names, so the
    name-swap comparison has a real signal to find."""
    import http.server
    import json as _json
    import threading

    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            text = (self.rfile.read(n) or b"").decode("utf-8", "replace")
            neg = ("Jamal" in text) or ("Lakisha" in text)
            content = (
                "We regret to inform the candidate that the profile is "
                "weak, not a fit, and we do not recommend proceeding."
                if neg
                else "The candidate is excellent, a strong fit, and we "
                "recommend proceeding to the interview stage."
            )
            resp = _json.dumps(
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
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        port = srv.server_address[1]
        d = run_pulse(
            pd.DataFrame(),
            {
                "source_kind": "endpoint",
                "llm_config": {
                    "endpoint_url": f"http://127.0.0.1:{port}/v1/chat/completions",
                    "api_format": "openai",
                    "model_name": "stub",
                    # The stub is on loopback; the egress guard refuses 127.0.0.1
                    # unless a caller opts in. Production callers must NOT set this.
                    "allow_loopback": True,
                },
            },
        )["data"]
    finally:
        srv.shutdown()
    assert d["sourceKind"] == "llm_endpoint"
    assert d["generative"]["available"] is True
    assert d["generative"]["perAxis"], "the probe produced no axis results"
    # Exact tone: a live-endpoint probe is a triage-level counterfactual
    # sample, so the verdict is always issued under a Disclaimer assurance
    # (reproduced deterministically against this stub on 2026-08-02). The
    # previous set-membership assertion listed the entire tone vocabulary
    # and could never fail.
    assert d["verdict"]["tone"] == "disclaimer"
    assert d["scope"]["pathway"] == "llm_endpoint"
