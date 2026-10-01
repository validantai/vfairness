"""Deterministic tests for the Pulse deferred tail (G-25 remainder,
G-38, G-43, G-44).

G-25 remainder: the counterfactual flip pass in scoring.py: a model
that takes a protected attribute as an input is re-scored with the
attribute flipped, and a changed decision reads as a counterfactual_flip
finding; a model that never sees the attribute reports the direct
pathway invariant by construction (notApplicable), never a fabricated
flip.

G-38: label-quality screen: recorded favorable-outcome base rate per
group (planted skew reads as label_baserate_skew) and inter-annotator
agreement by group when annotator columns exist (planted per-group
disagreement reads as annotator_disagreement); honest notApplicable
without ground truth.

G-43: memory-contamination screen on the trace contract's reserved
memory_reads/memory_writes fields: planted reliance disparity reads as
memory_reliance_disparity plus the timestamp-ordered cross-group
carry-over exposure; honest notApplicable when the traces carry no
memory events.

G-44: ethical-stance tags on metric cards (WAE vs WYSIWYG worldview),
the protected-encoding audit disclosure in dataPreparation, and the
first-vs-third-person fairness declaration on the live-endpoint probe.

Every test runs locally on in-memory frames or a stdlib http.server
stub: deterministic, no network, no external services.
"""

import http.server
import json
import threading

import numpy as np
import pandas as pd

from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.llm_probe import llm_probe_pulse
from vfairness.operations.pulse.scoring import score_model_over_frame

# ── G-25 remainder: counterfactual flip pass (scoring unit level) ─────────


def _flip_frame(n_per=100):
    rng = np.random.default_rng(11)
    n = 2 * n_per
    return pd.DataFrame(
        {
            "gender": ["A"] * n_per + ["B"] * n_per,
            "income": rng.normal(50.0, 10.0, n),
        }
    )


def _build_predict_on_gender(scoring_payload, model_cols):
    """Stub factory: the decision is PURELY the gender=B indicator."""
    idx = model_cols.index("gender=B")

    def predict(X):
        return (X[:, idx] >= 0.5).astype(float)

    return predict, "stub"


def test_flip_pass_detects_direct_protected_dependence():
    df = _flip_frame()
    inputs = {"protected_attributes": ["gender"], "feature_columns": ["income", "gender"]}
    out, feature_columns, _notes, degr = score_model_over_frame(
        df, inputs, {"model_base64": "ZmFrZQ=="}, _build_predict_on_gender
    )
    assert "prediction" in out.columns
    assert feature_columns == ["income", "gender"]
    assert not degr

    cf = inputs["_counterfactual_flips"]
    assert cf["available"] is True
    assert cf["modelCallsAdded"] == 1
    row = cf["perAttribute"][0]
    assert row["attribute"] == "gender"
    assert row["scoreScale"] == "binary"
    # The decision is purely the protected attribute: every row flips.
    assert row["flipRate"] == 1.0
    assert sorted(row["levels"]) == ["A", "B"]


def test_flip_pass_not_applicable_when_model_never_sees_protected():
    df = _flip_frame()
    inputs = {"protected_attributes": ["gender"], "feature_columns": ["income"]}

    def build_predict(scoring_payload, model_cols):
        def predict(X):
            return (X[:, 0] >= 50.0).astype(float)

        return predict, "stub"

    score_model_over_frame(df, inputs, {"model_base64": "ZmFrZQ=="}, build_predict)
    cf = inputs["_counterfactual_flips"]
    assert cf["available"] is False
    assert cf["notApplicable"] is True
    assert "does not take any protected attribute" in cf["reason"]


def test_flip_pass_scans_multi_level_attribute_per_level():
    """Three-level attribute: one model call per level; a model that
    ignores the attribute reads invariant (flipRate 0), and a model
    keyed on one level reads the flips plus the worst level pair."""
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east"], n),
            "income": rng.normal(50.0, 10.0, n),
        }
    )

    # (a) model ignores region entirely -> invariant across the scan.
    inputs = {"protected_attributes": ["region"], "feature_columns": ["income", "region"]}

    def build_predict_ignores(scoring_payload, model_cols):
        def predict(X):
            return (X[:, 0] >= 50.0).astype(float)

        return predict, "stub"

    score_model_over_frame(df, inputs, {"model_base64": "ZmFrZQ=="}, build_predict_ignores)
    cf = inputs["_counterfactual_flips"]
    assert cf["available"] is True
    row = cf["perAttribute"][0]
    assert row["attribute"] == "region"
    assert row["modelCalls"] == 3
    assert row["flipRate"] == 0.0
    assert cf["modelCallsAdded"] == 3

    # (b) model keyed purely on region=north -> every row's decision
    # differs between the north assignment and the others.
    inputs_b = {"protected_attributes": ["region"], "feature_columns": ["income", "region"]}

    def build_predict_north(scoring_payload, model_cols):
        idx = model_cols.index("region=north")

        def predict(X):
            return (X[:, idx] >= 0.5).astype(float)

        return predict, "stub"

    score_model_over_frame(df, inputs_b, {"model_base64": "ZmFrZQ=="}, build_predict_north)
    row_b = inputs_b["_counterfactual_flips"]["perAttribute"][0]
    assert row_b["flipRate"] == 1.0
    assert "north" in (row_b["worstPair"] or [])
    assert row_b["worstPairFlipRate"] == 1.0


def test_flip_pass_caps_level_scan_honestly():
    rng = np.random.default_rng(6)
    n = 160
    df = pd.DataFrame(
        {
            "origin": rng.choice([f"country_{i}" for i in range(8)], n),
            "income": rng.normal(50.0, 10.0, n),
        }
    )
    inputs = {"protected_attributes": ["origin"], "feature_columns": ["income", "origin"]}

    def build_predict(scoring_payload, model_cols):
        def predict(X):
            return (X[:, 0] >= 50.0).astype(float)

        return predict, "stub"

    score_model_over_frame(df, inputs, {"model_base64": "ZmFrZQ=="}, build_predict)
    cf = inputs["_counterfactual_flips"]
    assert cf["available"] is False
    assert cf["skipped"] and cf["skipped"][0]["attribute"] == "origin"
    assert "capped at 6" in cf["skipped"][0]["reason"]
    # No wasted model calls on a skipped scan.
    assert cf["modelCallsAdded"] == 0


# ── G-25 remainder: orchestrator integration ──────────────────────────────


def _tabular_frame(n_per=120):
    rng = np.random.default_rng(7)
    n = 2 * n_per
    return pd.DataFrame(
        {
            "gender": ["A"] * n_per + ["B"] * n_per,
            "prediction": rng.binomial(1, 0.5, n),
            "income": rng.normal(50.0, 10.0, n),
            "tenure": rng.normal(5.0, 2.0, n),
        }
    )


def test_orchestrator_attaches_flip_block_and_finding():
    df = _tabular_frame()
    inputs = {
        "protected_attributes": ["gender"],
        "model_base64": "ZmFrZQ==",
        "_counterfactual_flips": {
            "available": True,
            "perAttribute": [
                {
                    "attribute": "gender",
                    "levels": ["A", "B"],
                    "nRows": 240,
                    "scoreScale": "binary",
                    "flipRate": 0.5,
                    "meanAbsDelta": 0.5,
                    "maxAbsDelta": 1.0,
                }
            ],
            "skipped": [],
            "modelCallsAdded": 1,
            "method": "test",
        },
    }
    d = run_pulse(df, inputs)["data"]
    cf = d["individualFairness"]["counterfactual"]
    assert cf["available"] is True

    types = {f["type"]: f for f in d["bias"]}
    assert "counterfactual_flip" in types
    # 0.5 is far above the documented 5% critical threshold.
    assert types["counterfactual_flip"]["severity"] == "critical"
    assert "Flipping 'gender'" in types["counterfactual_flip"]["plain"]
    assert any("Counterfactual flip test" in c for c in d["scope"]["covered"])
    assert not any("Counterfactual individual fairness" in c for c in d["scope"]["notCovered"])


def test_orchestrator_names_flip_not_covered_when_pass_did_not_run():
    df = _tabular_frame()
    d = run_pulse(df, {"protected_attributes": ["gender"], "model_base64": "ZmFrZQ=="})["data"]
    assert "counterfactual" not in d["individualFairness"]
    assert any("Counterfactual individual fairness" in c for c in d["scope"]["notCovered"])
    assert not any("Counterfactual flip test" in c for c in d["scope"]["covered"])


# ── G-38: label-quality screen ─────────────────────────────────────────────


def _skewed_label_frame(n_a=200, n_b=100):
    """Planted label skew: group A carries the favorable label at 60%,
    group B at 10% (ratio 0.17, far under the 0.5 critical line).
    Predictions are label-independent noise."""
    rng = np.random.default_rng(13)
    n = n_a + n_b
    label = np.concatenate([rng.binomial(1, 0.60, n_a), rng.binomial(1, 0.10, n_b)])
    return pd.DataFrame(
        {
            "group": ["A"] * n_a + ["B"] * n_b,
            "prediction": rng.binomial(1, 0.5, n),
            "label": label,
            "feat": rng.normal(size=n),
        }
    )


def test_label_quality_flags_planted_baserate_skew():
    d = run_pulse(_skewed_label_frame(), {"protected_attributes": ["group"]})["data"]
    lq = d["labelQuality"]
    assert lq["available"] is True
    block = lq["labelBaseRates"][0]
    assert block["attribute"] == "group"
    assert block["referenceGroup"] == "A"
    assert block["worstGroup"] == "B"
    assert block["worstRatio"] < 0.5

    types = {f["type"]: f for f in d["bias"]}
    assert "label_baserate_skew" in types
    assert types["label_baserate_skew"]["severity"] == "critical"
    assert "RECORDED outcomes" in types["label_baserate_skew"]["plain"]
    assert any("Label-quality screen" in c for c in d["scope"]["covered"])
    # Honest framing: never claims to know WHY the labels skew.
    assert "cannot distinguish" in types["label_baserate_skew"]["plain"]


def test_label_quality_annotator_agreement_by_group():
    """Two annotator columns agree 100% on group A rows and only ~50%
    on group B rows: the agreement gap must be flagged for B."""
    rng = np.random.default_rng(17)
    n_per = 120
    n = 2 * n_per
    base = rng.binomial(1, 0.5, n)
    ann2 = base.copy()
    b_rows = np.arange(n_per, n)
    disagree = rng.choice(b_rows, size=n_per // 2, replace=False)
    ann2[disagree] = 1 - ann2[disagree]
    df = pd.DataFrame(
        {
            "group": ["A"] * n_per + ["B"] * n_per,
            "prediction": rng.binomial(1, 0.5, n),
            "label": base,
            "annotator_1": base,
            "annotator_2": ann2,
            "feat": rng.normal(size=n),
        }
    )
    d = run_pulse(df, {"protected_attributes": ["group"]})["data"]
    agr = d["labelQuality"]["annotatorAgreement"]
    assert agr["available"] is True
    assert sorted(agr["annotatorColumns"]) == ["annotator_1", "annotator_2"]
    assert 0.0 <= agr["overallKappa"] <= 1.0
    by_attr = {b["attribute"]: b for b in agr["byGroup"]}
    rows = {r["group"]: r for r in by_attr["group"]["groups"]}
    assert rows["A"]["agreement"] > 0.95
    assert rows["B"]["agreement"] < 0.60

    types = {f["type"]: f for f in d["bias"]}
    assert "annotator_disagreement" in types
    assert "least reliable" in types["annotator_disagreement"]["plain"]


def test_label_quality_not_applicable_without_ground_truth():
    rng = np.random.default_rng(19)
    n = 200
    df = pd.DataFrame(
        {
            "group": rng.choice(["A", "B"], n),
            "prediction": rng.binomial(1, 0.5, n),
            "feat": rng.normal(size=n),
        }
    )
    d = run_pulse(df, {"protected_attributes": ["group"]})["data"]
    lq = d["labelQuality"]
    assert lq["available"] is False
    assert lq["notApplicable"] is True
    assert "no real outcome column" in lq["reason"]
    assert not [
        f for f in d["bias"] if f["type"] in ("label_baserate_skew", "annotator_disagreement")
    ]


# ── G-44: stance tags + encoding audit ─────────────────────────────────────


def test_metric_cards_carry_ethical_stance_tags():
    d = run_pulse(_skewed_label_frame(), {"protected_attributes": ["group"]})["data"]
    by_metric = {}
    for card in d["metrics"]:
        by_metric.setdefault(card["metric"], card)
    dp = by_metric["demographic_parity_difference"]["stance"]
    assert dp["basis"] == "group" and dp["worldview"] == "WAE"
    eo = by_metric["equalized_odds_difference"]["stance"]
    assert eo["basis"] == "group" and eo["worldview"] == "WYSIWYG"
    assert "labels" in eo["note"]
    # The individual-fairness block self-tags as individual-basis.
    inf = d["individualFairness"]
    if inf.get("available"):
        assert inf["stance"]["basis"] == "individual"


def test_encoding_audit_names_binary_protected_encodings():
    rng = np.random.default_rng(23)
    n = 200
    df = pd.DataFrame(
        {
            "gender": rng.choice(["male", "female"], n),
            "region": rng.choice(["north", "south", "east"], n),
            "prediction": rng.binomial(1, 0.5, n),
            "feat": rng.normal(size=n),
        }
    )
    d = run_pulse(df, {"protected_attributes": ["gender", "region"]})["data"]
    audit = d["dataPreparation"]["encodingAudit"]
    attrs = [a["attribute"] for a in audit]
    assert "gender" in attrs
    assert "region" not in attrs, "three levels is not a binary encoding"
    gender_note = next(a for a in audit if a["attribute"] == "gender")
    assert "Non-binary people are invisible" in gender_note["detail"]


# ── G-44: first-vs-third-person declaration on the endpoint probe ─────────


_POS = (
    "The profile is excellent and the fit is good so this is an "
    "outstanding candidate and we are pleased to recommend proceeding"
)


def test_endpoint_probe_declares_third_person_perspective():
    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            resp = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": _POS}, "finish_reason": "stop"}
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
    try:
        port = srv.server_address[1]
        out = llm_probe_pulse(
            {
                "endpoint_url": "http://127.0.0.1:{0}/v1/chat/completions".format(port),
                "api_format": "openai",
                "model_name": "stub",
                "n_runs": 2,
                # The stub is on loopback; the egress guard refuses 127.0.0.1 unless
                # a caller opts in explicitly. Production callers must NOT set this.
                "allow_loopback": True,
            }
        )
    finally:
        srv.shutdown()
    assert out.get("success") is True
    d = out["data"]
    fp = d["generative"]["protocol"]["fairnessPerspective"]
    assert fp["measured"] == "third_person"
    assert fp["notMeasured"] == "first_person"
    assert any("First-person fairness" in c for c in d["scope"]["notCovered"])


# ── G-43: memory-contamination screen ──────────────────────────────────────


def _memory_traces(n_per=30, with_memory=True):
    """Planted reliance disparity: group A episodes read memory in 80%
    of cases, group B in 10%. Writes exist on both sides so the
    carry-over exposure surface is non-trivial; timestamps interleave
    the groups deterministically."""
    rng = np.random.default_rng(29)
    n = 2 * n_per
    groups = (["A", "B"] * n_per)[:n]
    reads = [
        (
            rng.integers(1, 4)
            if ((g == "A" and rng.random() < 0.8) or (g == "B" and rng.random() < 0.1))
            else 0
        )
        for g in groups
    ]
    writes = [int(rng.random() < 0.3) for _ in range(n)]
    cols = {
        "trace_id": ["t{0}".format(i) for i in range(n)],
        "group": groups,
        "tool": rng.choice(["search", "calc"], n),
        "timestamp": np.arange(n, dtype=float),
    }
    if with_memory:
        cols["memory_reads"] = reads
        cols["memory_writes"] = writes
    return pd.DataFrame(cols)


def test_memory_screen_flags_planted_reliance_disparity():
    d = run_pulse(_memory_traces(), {"protected_attributes": ["group"], "source_kind": "agent"})[
        "data"
    ]
    assert d["sourceKind"] == "agent_traces"
    mem = d["agent"]["memory"]
    assert mem["available"] is True
    rows = {r["group"]: r for r in mem["perGroup"]}
    assert rows["A"]["episodesReadingShare"] > 0.6
    assert rows["B"]["episodesReadingShare"] < 0.3
    assert mem["relianceChi2P"] < 0.05
    assert mem["exposure"]["available"] is True
    assert mem["exposure"]["timeColumn"] == "timestamp"
    # Honest boundary: this fixture carries counts but no payload text,
    # so the term screen did not run and the boundary is named.
    assert mem["contentAudited"] is False
    assert "no memory payload text" in mem["notCovered"]

    types = {f["type"]: f for f in d["bias"]}
    assert "memory_reliance_disparity" in types
    assert "memory-reliant group" in types["memory_reliance_disparity"]["plain"]
    assert any("Memory-contamination screen" in c for c in d["scope"]["covered"])


def test_memory_term_screen_flags_cross_group_descriptors():
    """Per-episode table with memory payload text: episodes of group
    'candidate_b' read memory containing the OTHER group's descriptor
    ('candidate_a'), which must fire memory_term_contamination; the
    reverse direction stays clean."""
    rng = np.random.default_rng(31)
    n_per = 25
    n = 2 * n_per
    groups = ["candidate_a"] * n_per + ["candidate_b"] * n_per
    read_text = (
        [""] * n_per  # group a reads nothing informative
        + ["stored note: candidate_a prefers aggressive follow-up"]
        * n_per  # group b reads a-descriptor payloads
    )
    df = pd.DataFrame(
        {
            "trace_id": ["t{0}".format(i) for i in range(n)],
            "group": groups,
            "tool": rng.choice(["search", "calc"], n),
            "timestamp": np.arange(n, dtype=float),
            "memory_reads": [1] * n,
            "memory_writes": [0] * n_per + [1] * n_per,
            "memory_read_text": read_text,
            "memory_write_text": [""] * n,
        }
    )
    d = run_pulse(df, {"protected_attributes": ["group"], "source_kind": "agent"})["data"]
    mem = d["agent"]["memory"]
    assert mem["available"] is True
    assert mem["contentAudited"] is True
    ts = mem["termScreen"]
    assert ts["available"] is True
    rows = {r["group"]: r for r in ts["perGroup"]}
    assert rows["candidate_b"]["crossGroupDescriptorShare"] == 1.0
    assert rows["candidate_b"]["descriptorsSeen"] == ["candidate_a"]
    assert rows["candidate_a"]["crossGroupDescriptorShare"] == 0.0

    types = {f["type"]: f for f in d["bias"]}
    assert "memory_term_contamination" in types
    # A descriptor-match screen is surfaced for review, not a significance
    # test, so it is capped at "warn" and cannot block deployment on its
    # own (audit fix mem-1). The finding still fires and names the match.
    assert types["memory_term_contamination"]["severity"] == "warn"
    assert types["memory_term_contamination"]["statisticalTest"] is None
    assert "descriptor" in types["memory_term_contamination"]["plain"]
    # Honest boundary: not a general PII sweep.
    assert "observed group descriptors" in mem["notCovered"]
    assert any("term-level descriptor scan" in c for c in d["scope"]["covered"])


def test_span_flattener_retains_bounded_memory_payloads():
    """OTel-style span export with memory read/write spans carrying
    content attributes: the flattened table must carry the payload text
    per episode, bounded, and NaN when no memory events exist."""
    from vfairness.operations.pulse.traces import parse_trace_export

    spans = []
    for i in range(3):
        tid = "trace_{0}".format(i)
        spans.append(
            {
                "trace_id": tid,
                "name": "agent.step",
                "attributes": {
                    "group": "g_one" if i < 2 else "g_two",
                    "gen_ai.tool.name": "search",
                    "start_time": 1000 + i,
                },
            }
        )
        spans.append(
            {
                "trace_id": tid,
                "name": "memory.retrieve",
                "attributes": {"content": "profile note {0}".format(i), "start_time": 1001 + i},
            }
        )
        spans.append(
            {
                "trace_id": tid,
                "name": "memory.store",
                "attributes": {"content": "x" * 5000, "start_time": 1002 + i},
            }
        )
    df = parse_trace_export(spans)
    assert "memory_read_text" in df.columns
    row = df[df["trace_id"] == "trace_0"].iloc[0]
    assert "profile note 0" in row["memory_read_text"]
    # The write payload is clipped to the documented per-direction cap.
    assert len(row["memory_write_text"]) <= 4000

    # No memory events at all -> NaN, never an empty-string fabrication.
    plain = [
        {
            "trace_id": "t",
            "name": "agent.step",
            "attributes": {"group": "g", "gen_ai.tool.name": "calc", "start_time": 1.0},
        }
    ]
    df2 = parse_trace_export(plain)
    assert df2["memory_read_text"].isna().all()


def test_memory_screen_not_applicable_without_memory_events():
    d = run_pulse(
        _memory_traces(with_memory=False),
        {"protected_attributes": ["group"], "source_kind": "agent"},
    )["data"]
    mem = d["agent"]["memory"]
    assert mem["available"] is False
    assert mem["notApplicable"] is True
    assert "no memory read/write events" in mem["reason"]
    assert not [f for f in d["bias"] if f["type"] == "memory_reliance_disparity"]
    assert any("Memory-contamination screening" in c for c in d["scope"]["notCovered"])


# ── G-16: vision sidecar disclosure on the engine side ─────────────────────


def test_vision_path_discloses_sidecar_classifier_when_present():
    rng = np.random.default_rng(37)
    n = 60
    df = pd.DataFrame(
        {
            "image_url": ["https://x.test/i{0}.jpg".format(i) for i in range(n)],
            "detected_gender": rng.choice(
                ["male-presenting", "female-presenting"], n, p=[0.8, 0.2]
            ),
        }
    )
    d = run_pulse(
        df,
        {
            "protected_attributes": [],
            "source_kind": "image",
            "_vision_classifier": {
                "model": "openai/clip-vit-base-patch32",
                "axis": "perceived_gender",
                "labelColumn": "detected_gender",
                "nClassified": n,
                "nFailed": 3,
                "nDropped": 3,
                "meanConfidence": 0.91,
            },
        },
    )["data"]
    assert d["sourceKind"] == "image_set"
    v = d["vision"]
    assert v["labelsSource"] == "classified_by_sidecar_clip_zero_shot"
    assert v["classifier"]["model"] == "openai/clip-vit-base-patch32"
    assert v["classifier"]["nFailed"] == 3
    assert any("vision sidecar" in c for c in d["scope"]["covered"])
    # The classifier's own error skew stays honestly not-covered.
    assert any("OWN demographic error skew" in c for c in d["scope"]["notCovered"])


def test_vision_path_unchanged_without_sidecar():
    rng = np.random.default_rng(38)
    df = pd.DataFrame(
        {
            "detected_race": rng.choice(["White", "Black"], 50, p=[0.9, 0.1]),
        }
    )
    d = run_pulse(df, {"protected_attributes": [], "source_kind": "image"})["data"]
    v = d["vision"]
    assert v["labelsSource"] == "perceived_by_classifier"
    assert v["classifier"] is None
    assert any("no vision sidecar ran" in c for c in d["scope"]["notCovered"])


# ── Audit 2026-07-11: the significance gate on the verdict roll-up ─────────


def test_magnitude_only_critical_does_not_flip_verdict_to_adverse():
    """A magnitude-only bias finding (statisticalTest None, no significant
    flag) must be surfaced but capped at 'warn' for the overall opinion,
    so small-sample noise cannot produce Adverse / blocksDeployment. A
    statistically confirmed critical still escalates."""
    from vfairness.operations.reporting import build_assurance_verdict

    unconfirmed = build_assurance_verdict(
        schema={"pii_leakage": [], "mismatches": [], "refuse": False},
        per_variable=[{"attribute": "g", "assessable": True, "gap": 0.0, "significant": False}],
        metrics=[],
        proxies={},
        statistical={},
        intersectional={},
        disparity_matrix={},
        bias=[
            {
                "type": "calibration_disparity",
                "severity": "critical",
                "plain": "ECE gap 0.11",
                "statisticalTest": None,
            }
        ],
        recommended={},
        domain="hiring",
        jurisdiction="eu",
        has_truth=True,
    )
    assert unconfirmed["overall"] != "Adverse"
    assert unconfirmed["blocksDeployment"] is False

    confirmed = build_assurance_verdict(
        schema={"pii_leakage": [], "mismatches": [], "refuse": False},
        per_variable=[{"attribute": "g", "assessable": True, "gap": 0.0, "significant": False}],
        metrics=[],
        proxies={},
        statistical={},
        intersectional={},
        disparity_matrix={},
        bias=[
            {
                "type": "cohort_error_concentration",
                "severity": "critical",
                "plain": "3x baseline, binomial p 0.001",
                "significant": True,
                "statisticalTest": {"test": "binomial_vs_baseline", "pValue": 0.001},
            }
        ],
        recommended={},
        domain="hiring",
        jurisdiction="eu",
        has_truth=True,
    )
    assert confirmed["overall"] == "Adverse"
    assert confirmed["blocksDeployment"] is True
