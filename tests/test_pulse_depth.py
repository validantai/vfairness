"""Deterministic tests for Pulse close-plan Phase 6 (G-23, G-25, G-35,
G-42 stretch).

G-23: refusal-rate disparity on the live-endpoint probe (per-arm refusal
rates, Fisher tests folded into the SAME Benjamini-Hochberg family as
the sentiment cells) plus the explicit quality-of-service block on the
prompts+outputs path (refusalRateByGroup, dimensionsCovered, the
toxicity caveat).

G-25: individual fairness on the tabular binary path: kNN decision
consistency (k=10) over the non-protected features plus a Theil-T
between/within-group inequality decomposition, with honest
notApplicable below 100 rows.

G-35: depth-2 error-cohort analysis (Azure error-tree style), only
against real outcomes; a planted bad cohort is localized and flagged.

G-42: trimmed-mean divergence robustnessNote on perVariable rows when
extreme-score rows carry part of the headline gap.

Every test runs locally against a stdlib http.server stub or a pure
in-memory frame: deterministic, no network, no external services.
"""

import http.server
import json
import threading

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.llm_probe import llm_probe_pulse

# Bare sentiment keywords (no punctuation glued onto them) so the
# built-in keyword sentiment scorer separates arms deterministically.
_POS = (
    "The profile is excellent and the fit is good so this is an "
    "outstanding candidate and we are pleased to recommend proceeding"
)

# Clear hard refusal ("i cannot" pattern) with plenty of English function
# words so the English-marker guard never trips on a refusal corpus.
_REFUSE = (
    "I am sorry but I cannot help with this request because it is "
    "against my guidelines and I will not proceed with the "
    "assessment of this application"
)


# ── stub plumbing (same pattern as tests/test_pulse_llm_probe.py) ─────────


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


# ── G-23 (probe): refusal-rate disparity across counterfactual arms ───────


def test_probe_refusal_disparity_fires_family_wise():
    """Stub refuses ONLY for Jamal and answers everyone else. The name
    axis must carry per-arm refusal rates (Jamal 1.0, all others 0.0),
    and a refusal finding must fire under the SAME family-wise BH regime
    the sentiment cells use, naming the quality-of-service harm."""

    def decide(body):
        return _REFUSE if "Jamal" in body else _POS

    d = _probe_against(decide)
    rows = {r["axis"]: r for r in d["generative"]["perAxis"]}

    # Per-arm refusal rates on the name axis.
    rr = rows["name"]["refusalRates"]
    assert rr["Jamal"] == 1.0
    for arm in ("Greg", "Emily", "Lakisha", "Mei", "Mohammed"):
        assert rr[arm] == 0.0, arm
    # The descriptor axes saw only answered prompts: rates all zero.
    for axis in ("age", "disability", "religion"):
        assert rows[axis]["refusalRates"], axis
        assert all(v == 0.0 for v in rows[axis]["refusalRates"].values())

    # Exactly the Jamal comparison is family-wise significant.
    comps = {c["arm"]: c for c in rows["name"]["refusalComparisons"]}
    assert comps["Jamal"]["familyWiseSignificant"] is True
    assert comps["Jamal"]["pValue"] <= 0.05
    # LF-20, 2026-09-10. This asserted `is False` for the other four arms, and
    # False there meant "tested, and no disparity". They were never tested: the
    # reference arm and each of them refused NOTHING on every shared template,
    # so an exact McNemar has zero discordant pairs and therefore no evidence at
    # all. Fisher answered 1.0 for them, and those four p=1.0 entries joined the
    # Benjamini-Hochberg family as full members with zero power, which is the
    # defect READINESS-6 removed from nine other detectors. They are now None,
    # which keeps them out of the family and makes a real finding easier to see.
    for arm in ("Emily", "Lakisha", "Mei", "Mohammed"):
        assert comps[arm]["pValue"] is None, arm
        assert comps[arm]["familyWiseSignificant"] is None, arm
        assert comps[arm]["discordantTemplates"] == 0, arm
        assert comps[arm]["detectable"] is False, arm
    # The paired test is strictly more conservative here, and that is the point:
    # McNemar 0.03125 against Fisher 0.0022 on the same total split. The
    # unpaired number is kept and labelled so the change is legible in a diff of
    # two runs.
    assert comps["Jamal"]["method"] == "mcnemar_exact"
    assert comps["Jamal"]["discordantTemplates"] == comps["Jamal"]["pairedTemplates"]
    assert comps["Jamal"]["pValue"] > comps["Jamal"]["unpairedFisherPValue"]
    assert "no verdict is taken from it" in comps["Jamal"]["unpairedNote"]

    # The refusal finding: quality-of-service framing, computed severity
    # (a 100-point refusal gap is a categorical denial of service).
    ref = [f for f in d["bias"] if f["type"] == "llm_refusal_rate_disparity"]
    assert len(ref) == 1
    f = ref[0]
    assert f["attribute"] == "name"
    assert f["severity"] == "critical"
    assert f["qualityOfService"] is True
    assert "quality-of-service" in f["plain"]
    assert "Jamal" in f["plain"]
    assert f["refusalRates"]["Jamal"] == 1.0
    st = f["statisticalTest"]
    assert st["familyWiseCorrection"] == "benjamini_hochberg"
    assert st["minPValue"] <= 0.05
    # LF-20: the disclosure names the test that was actually run, and says why
    # it is the paired one.
    assert "McNemar" in st["method"]
    assert "matched pair" in st["method"]
    assert "Fisher" not in st["method"]


def test_probe_no_refusals_means_no_refusal_finding():
    """An endpoint that answers every arm identically must not produce a
    refusal finding, and every arm's refusal rate reads 0.0."""

    def decide(body):
        return _POS

    d = _probe_against(decide)
    assert not [f for f in d["bias"] if f["type"] == "llm_refusal_rate_disparity"]
    for row in d["generative"]["perAxis"]:
        assert all(v == 0.0 for v in row["refusalRates"].values())


# ── G-23 (prompts+outputs): qualityOfService + dimensionsCovered ──────────


def _qos_df():
    """Group X's outputs are mostly refusal-worded (12 of 15); group Y is
    always answered. Mixed X texts keep within-group variance non-zero so
    the refusal_rate Mann-Whitney comparison is well defined."""
    rows = [
        {"prompt": "p", "response": _REFUSE + " case {0}.".format(i), "group": "X"}
        for i in range(12)
    ]
    rows += [
        {"prompt": "p", "response": _POS + " case {0}.".format(i), "group": "X"} for i in range(3)
    ]
    rows += [
        {"prompt": "p", "response": _POS + " case {0}.".format(i), "group": "Y"} for i in range(15)
    ]
    return pd.DataFrame(rows)


def test_generative_quality_of_service_block():
    d = run_pulse(_qos_df(), {"source_kind": "prompts_outputs", "protected_attributes": ["group"]})[
        "data"
    ]
    gen = d["generative"]
    assert gen["available"] is True

    # dimensionsCovered names every metric that actually scored.
    dims = gen["dimensionsCovered"]
    assert set(dims) >= {
        "sentiment",
        "toxicity",
        "refusal_rate",
        "regard",
        "helpfulness",
        "stereotype",
        "semantic_quality",
        "information_quality",
        "representation",
        "framing",
        "response_length",
    }

    qos = gen["qualityOfService"]
    rates = qos["refusalRateByGroup"]["group"]
    assert abs(rates["X"] - 0.8) < 1e-9
    assert rates["Y"] == 0.0
    assert "over-flag identity terms" in qos["toxicityCaveat"]
    assert "leads, not verdicts" in qos["toxicityCaveat"]

    # The refusal-rate gap joined the family-wise FDR family and fired a
    # finding carrying the quality-of-service framing.
    qf = [f for f in d["bias"] if f.get("qualityOfService")]
    assert qf, "refusal-rate disparity must fire a finding"
    assert any("quality-of-service" in f["plain"] for f in qf)
    assert any("refusal_rate" in f["plain"] for f in qf)


# ── G-25: individual fairness (kNN consistency + Theil) ───────────────────


def _inconsistent_df(n_per=150):
    """Planted individual-fairness violation: both groups draw features
    from the SAME distribution, but the decision is purely the group
    (G1 always selected, G2 never). Any point's nearest neighbours on
    the features are a ~50/50 group mix, so kNN agreement collapses to
    ~0.5 and the Theil between-group share saturates near 1.0."""
    rng = np.random.default_rng(21)
    n = 2 * n_per
    return pd.DataFrame(
        {
            "group": ["G1"] * n_per + ["G2"] * n_per,
            "prediction": [1] * n_per + [0] * n_per,
            "feat_a": rng.normal(0.0, 1.0, n),
            "feat_b": rng.normal(0.0, 1.0, n),
        }
    )


def test_individual_fairness_planted_inconsistency():
    d = run_pulse(_inconsistent_df(), {"protected_attributes": ["group"]})["data"]
    inf = d["individualFairness"]
    assert inf["available"] is True
    assert inf["k"] == 10
    assert inf["consistencyScore"] < 0.65, (
        "group-blind neighbours must disagree about half the time"
    )
    pg = inf["perGroupConsistency"]["group"]
    assert pg["G1"] < 0.65 and pg["G2"] < 0.65

    th = inf["theil"]
    assert th["attribute"] == "group"
    assert th["total"] > 0
    assert th["betweenShare"] > 0.95, "outcome is purely group-driven"
    assert abs(th["betweenGroups"] + th["withinGroups"] - th["total"]) < 1e-9
    # Documented thresholds are disclosed on the payload.
    assert inf["thresholds"]["consistencyFloor"] == 0.80
    assert inf["thresholds"]["betweenShareCeiling"] == 0.20
    assert "plain" in inf and inf["plain"]

    types = {f["type"]: f for f in d["bias"]}
    assert "individual_fairness" in types
    assert types["individual_fairness"]["severity"] == "critical"
    assert "Similar individuals" in types["individual_fairness"]["plain"]
    assert "inequality_index" in types
    assert types["inequality_index"]["severity"] == "critical"
    assert "Theil" in types["inequality_index"]["plain"]


def test_individual_fairness_not_applicable_below_100_rows():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "group": ["A"] * 25 + ["B"] * 25,
            "prediction": rng.binomial(1, 0.5, 50),
            "feat": rng.normal(size=50),
        }
    )
    d = run_pulse(df, {"protected_attributes": ["group"]})["data"]
    inf = d["individualFairness"]
    assert inf["available"] is False
    assert inf["notApplicable"] is True
    assert "100" in inf["reason"]
    # A skipped stage is not a degradation and fires no finding.
    assert not [f for f in d["bias"] if f["type"] in ("individual_fairness", "inequality_index")]
    assert "individual_fairness" not in [g.get("stage") for g in d["degradations"]]


# ── G-35: cohort error analysis ───────────────────────────────────────────


def _cohort_df(n=480, with_truth=True):
    """Planted bad cohort: rows with f1=b AND f2=y are misclassified at
    ~45% while everything else sits at ~5% (roughly 3x the pooled
    baseline). Group is decision-independent noise."""
    rng = np.random.default_rng(9)
    f1 = rng.choice(["a", "b"], n)
    f2 = rng.choice(["x", "y"], n)
    group = rng.choice(["A", "B"], n)
    label = rng.binomial(1, 0.5, n)
    bad = (f1 == "b") & (f2 == "y")
    flip = np.where(bad, rng.binomial(1, 0.45, n), rng.binomial(1, 0.05, n))
    pred = np.where(flip == 1, 1 - label, label)
    cols = {"group": group, "f1": f1, "f2": f2, "prediction": pred, "feature": rng.normal(size=n)}
    if with_truth:
        cols["label"] = label
    return pd.DataFrame(cols)


def test_cohort_analysis_localizes_planted_bad_cohort():
    d = run_pulse(_cohort_df(), {"protected_attributes": ["group"]})["data"]
    co = d["cohorts"]
    assert co["available"] is True
    assert 0.0 < co["baselineErrorRate"] < 0.25
    assert co["worst"], "worst cohorts must be listed"
    w0 = co["worst"][0]
    assert "f1=b" in w0["cohort"] and "f2=y" in w0["cohort"]
    assert w0["n"] >= 30
    assert w0["errorRate"] >= 2 * w0["baselineErrorRate"]
    assert w0["lift"] >= 2
    assert len(co["worst"]) <= 5
    assert "error-tree" in co["method"]

    finding = [f for f in d["bias"] if f["type"] == "cohort_error_concentration"]
    assert len(finding) == 1
    f = finding[0]
    assert f["severity"] in ("warn", "critical")
    assert "f1=b" in f["plain"] and "f2=y" in f["plain"]
    assert "specification or proxy bias" in f["plain"]


def test_cohort_analysis_unavailable_without_ground_truth():
    d = run_pulse(_cohort_df(with_truth=False), {"protected_attributes": ["group"]})["data"]
    co = d["cohorts"]
    assert co["available"] is False
    assert "ground-truth" in co["reason"]
    assert not [f for f in d["bias"] if f["type"] == "cohort_error_concentration"]
    # An honest skip is not a stage collapse.
    assert "cohort_analysis" not in [g.get("stage") for g in d["degradations"]]


# ── G-42 (stretch): trimmed-mean divergence robustnessNote ────────────────


def _outlier_df():
    """Group B's positives sit disproportionately in its extreme 5% of
    scores (both tails), group A's never do. Trimming each group's
    extreme scores moves the raw 10-point gap to ~15.8 points, a ~58%
    relative divergence, well past the documented 20% flag line."""
    n = 200
    s = np.linspace(0.0, 1.0, n)
    pred_a = np.zeros(n, dtype=int)
    pred_a[70:130] = 1  # 60 mid-score positives
    pred_b = np.zeros(n, dtype=int)
    pred_b[85:115] = 1  # 30 mid-score positives
    pred_b[:5] = 1  # 5 extreme-low positives
    pred_b[-5:] = 1  # 5 extreme-high positives
    return pd.DataFrame(
        {
            "group": ["A"] * n + ["B"] * n,
            "prediction": np.concatenate([pred_a, pred_b]),
            "score": np.concatenate([s, s]),
            "feat": np.tile(np.linspace(-1.0, 1.0, n), 2),
        }
    )


def test_trimmed_divergence_adds_robustness_note():
    d = run_pulse(_outlier_df(), {"protected_attributes": ["group"]})["data"]
    row = next(v for v in d["perVariable"] if v["attribute"] == "group")
    assert row["assessable"] is True
    assert abs(row["gap"] - 0.10) < 1e-9
    assert "trimmedGap" in row
    assert row["trimmedGap"] > row["gap"] * 1.2, (
        "trimming B's tail positives must widen the gap by > 20% relative"
    )
    note = row.get("robustnessNote") or ""
    assert "extreme 5 percent" in note
    assert "data-entry outliers" in note


# ── READINESS-5: the refusal detector must be able to fire on its own ──


def test_a_total_refusal_fires_without_help_from_the_sentiment_cells():
    """The refusal probe must not depend on the sentiment probe finding something.

    Until 2026-09-10 it did, invisibly. The refusal-rate comparisons shared one
    Benjamini-Hochberg family with every (axis, template) sentiment cell, and
    Fisher's exact is discrete: with the name axis's 6 templates per arm, the
    smallest p a PERFECT split can produce is 0.0021645, while the pooled
    family of 32 put the rank-1 threshold at 0.05/32 = 0.0015625. So a model
    refusing every single request for one name and none for any other could not
    be reported, however total the denial of service was.

    It looked like it worked, because six sentiment cells carried p=0.0 and
    lifted the refusal comparison to rank 7. Those p=0.0 values came from
    comparing a sentiment of 1.0 against a sentiment of 0.0 that the keyword
    scorer had INVENTED for a refusal message containing none of its lexicon
    words. Fixing that fabrication silenced the refusal detector, which is how
    the dependency was found.

    This test pins the independence directly: the sentiment side finds nothing
    here, and the refusal finding has to fire anyway.
    """
    d = _probe_against(lambda body: _REFUSE if "Jamal" in body else _POS)
    rows = {r["axis"]: r for r in d["generative"]["perAxis"]}
    name = rows["name"]

    # The sentiment side has nothing: every Jamal pair is unmeasurable (a
    # refusal message has no sentiment the lexicon can read) and every other
    # arm is identical to the reference.
    assert not any(c["familyWiseSignificant"] for c in name["perTemplate"]), (
        "a sentiment cell fired, so this no longer tests the independence"
    )
    jamal_pairs = [pr for c in name["perTemplate"] for pr in c["pairs"] if pr["arm"] == "Jamal"]
    assert jamal_pairs and all(pr["assessed"] is False for pr in jamal_pairs), (
        "the Jamal sentiment pairs were assessed; the fabricated 0.0 is back"
    )
    assert all(pr["pValue"] is None for pr in jamal_pairs), (
        "an unmeasurable pair carries a p-value, which would rejoin the family"
    )

    # And the refusal finding fires regardless.
    ref = [f for f in d["bias"] if f["type"] == "llm_refusal_rate_disparity"]
    assert len(ref) == 1, (
        "a model refusing 100% of requests for one name and 0% for every other "
        "produced no refusal finding"
    )
    assert ref[0]["severity"] == "critical"


def test_a_refusal_test_with_no_power_says_so_instead_of_passing():
    """Three states on the detector itself, not just on its result.

    "Not family-wise significant" is read as "this model serves every arm
    alike". For a small template set that reading is false: the test could not
    have said anything else. The exact McNemar is discrete and its evidence is
    only the DISCORDANT templates, so four of them cannot produce a p below
    0.125, and a family of five sets the rank-1 threshold at 0.01. A total
    refusal would be reported as clean.

    So each comparison carries whether it COULD have fired, and a design that
    could not says so in words rather than passing quietly.
    """
    from vfairness.operations.pulse.llm_probe import (
        _annotate_refusal_power,
        _min_attainable_refusal_p,
    )

    # The arithmetic this rests on, checked rather than assumed.
    assert _min_attainable_refusal_p(4, 4) == pytest.approx(0.0285714, rel=1e-4)
    assert _min_attainable_refusal_p(6, 6) == pytest.approx(0.0021645, rel=1e-4)

    # LF-20: the floor now belongs to the exact McNemar the path actually runs,
    # so a comparison has to carry its DISCORDANT count. That is the whole
    # point: 200 templates with 5 disagreements has the power of a 5-template
    # study, and the per-arm template count cannot say so.
    def annotated(n_templates, n_discordant=None):
        d = n_templates if n_discordant is None else n_discordant
        comps = [
            {
                "arm": f"arm{i}",
                "pValue": 1.0,
                "templatesTested": n_templates,
                "pairedTemplates": n_templates,
                "discordantTemplates": d,
            }
            for i in range(5)
        ]
        _annotate_refusal_power(comps)
        return comps

    # Four discordant templates: the threshold is 0.05/5 = 0.01 and the exact
    # McNemar floor is 0.125. Nothing this probe observes could ever be
    # reported.
    weak = annotated(4)
    assert all(c["detectable"] is False for c in weak)
    note = weak[0]["detectabilityNote"]
    assert "NOT DETECTABLE" in note
    assert "absence of statistical power" in note, (
        "the note has to say what a clean reading here does NOT mean"
    )
    assert "templates that BOTH arms treat the same way does not help" in note, (
        "the reader has to know that running more templates is not the fix"
    )

    # LF-20, AND THIS IS A REAL REDUCTION IN WHAT THE PROBE CAN CLAIM, recorded
    # here rather than tuned away. Six discordant templates against five
    # comparisons is NOT enough under the paired test: the McNemar floor is
    # 0.03125 and the rank-1 bar is 0.05/5 = 0.01. Under the old unpaired
    # Fisher reading, six templates per arm cleared it at 0.0022, so this
    # design used to look sound and no longer does. That is the anti-
    # conservatism the pairing removes, and the silence it replaces is worse
    # than the loss.
    still_weak = annotated(6)
    assert all(c["detectable"] is False for c in still_weak)

    # Over-correction control. A guard that marked everything undetectable
    # would be as useless as the silence it replaced, so a design that CAN fire
    # must come back clean: eight discordant templates floor at 0.0078, which
    # clears the same 0.01 bar.
    strong = annotated(20, n_discordant=8)
    assert all(c["detectable"] is True for c in strong)
    assert all(c["detectabilityNote"] == "" for c in strong)

    # And the family size is what moved, not the data: the same eight
    # discordant templates against ONE comparison clear 0.05 easily.
    solo = [{"arm": "a", "pValue": 0.0078, "templatesTested": 20, "discordantTemplates": 8}]
    _annotate_refusal_power(solo)
    assert solo[0]["detectable"] is True

    # Could-not-check stays its own state: an unmeasurable comparison is not
    # annotated as either detectable or undetectable.
    mixed = [
        {"arm": "a", "pValue": None, "templatesTested": 0, "discordantTemplates": None},
        {"arm": "b", "pValue": 1.0, "templatesTested": 6, "discordantTemplates": 6},
    ]
    _annotate_refusal_power(mixed)
    assert mixed[0]["detectable"] is None
    assert "COULD NOT CHECK" in mixed[0]["detectabilityNote"]
    assert mixed[1]["detectable"] is True
