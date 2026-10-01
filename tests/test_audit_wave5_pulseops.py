"""Audit wave 5 pins: Pulse + ops small correctness fixes.

One test per confirmed finding. All deterministic: seeded RNGs, stdlib
HTTP stub for the LLM probe, fake dowhy module for the refutation suite.
"""

import http.server
import inspect
import json
import math
import sys
import threading
import types
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

# ── experimentation/analysis: Baron-Kenny step 4 actually tested ──────────


def _mk_experiment(control, treatment, outcome="y"):
    return types.SimpleNamespace(control=control, treatment=treatment, outcome=outcome)


def test_mediation_step4_rejects_noise_mediator():
    """A pure-noise mediator must NOT satisfy step 4 (the b path). The old
    code hardcoded step4 True, certifying mediation through noise."""
    from vfairness.operations.experimentation.analysis import ExperimentAnalysis

    rng = np.random.default_rng(42)
    n = 400
    exp = _mk_experiment(
        pd.DataFrame({"y": rng.normal(0, 1, n), "m": rng.normal(0, 1, n)}),
        pd.DataFrame({"y": rng.normal(2, 1, n), "m": rng.normal(0, 1, n)}),
    )
    dec = ExperimentAnalysis(None, experiment=exp).mediation_analysis("m")
    assert dec.steps_satisfied["step4_mediator_significant"] is False


def test_mediation_step4_accepts_real_mediator():
    """Positive control: a genuine T -> M -> Y chain satisfies step 4."""
    from vfairness.operations.experimentation.analysis import ExperimentAnalysis

    rng = np.random.default_rng(7)
    n = 400
    m_c = rng.normal(0, 1, n)
    m_t = rng.normal(2, 1, n)
    exp = _mk_experiment(
        pd.DataFrame({"m": m_c, "y": 1.5 * m_c + rng.normal(0, 1, n)}),
        pd.DataFrame({"m": m_t, "y": 1.5 * m_t + rng.normal(0, 1, n)}),
    )
    dec = ExperimentAnalysis(None, experiment=exp).mediation_analysis("m")
    assert dec.steps_satisfied["step4_mediator_significant"] is True
    assert dec.proportion_mediated > 0.5


def test_ols_coef_matches_direct_formula():
    """Pin the _ols_coef output against a direct computation, so the
    removed dead y_hat line stays a pure no-op."""
    from vfairness.evaluation.vfairness_metrics._statistics import _norm_cdf
    from vfairness.operations.experimentation.analysis import ExperimentAnalysis

    rng = np.random.default_rng(3)
    x = rng.normal(size=50)
    y = 2 * x + rng.normal(size=50)
    beta, p = ExperimentAnalysis._ols_coef(x, y)
    beta0 = y.mean() - beta * x.mean()
    resid = y - (beta0 + beta * x)
    se = np.sqrt(np.sum(resid**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    p_direct = float(2 * _norm_cdf(-abs(beta / se)))
    assert beta == pytest.approx(2.0, abs=0.3)
    assert p == pytest.approx(p_direct, abs=1e-12)


# ── pulse/orchestrator: min-group docs match the code default (20) ────────


def test_min_group_docs_match_code_default():
    from vfairness.operations.pulse import orchestrator as orch

    assert orch._MIN_GROUP_DEFAULT == 20
    doc = orch._resolve_min_group.__doc__ or ""
    assert "default is 20" in doc
    assert "default is 30" not in doc
    # No-override call resolves to the documented default.
    assert orch._resolve_min_group({}) == 20


# ── pulse/regulatory: honest note for unparseable target_ratio ─────────────


def test_unparseable_target_ratio_note_says_default_used():
    from vfairness.operations.pulse.regulatory import build_targets

    t = build_targets([], "not-a-number")
    assert t["targetRatio"] == 0.8
    assert "could not be parsed" in t["note"]
    assert "clamped" not in t["note"]


def test_out_of_band_target_ratio_still_says_clamped():
    from vfairness.operations.pulse.regulatory import build_targets

    t = build_targets([], 1.7)
    assert t["targetRatio"] == 1.0
    assert "clamped" in t["note"]
    # In-band value: neither note.
    ok = build_targets([], 0.9)
    assert "clamped" not in ok["note"]
    assert "could not be parsed" not in ok["note"]


# ── pulse/traces: 'set' (and 'persist') classify as memory writes ─────────


def test_memory_set_and_persist_are_writes():
    from vfairness.operations.pulse.traces import _memory_kind

    assert _memory_kind("memory.set", None) == "write"
    assert _memory_kind("mem0.persist", None) == "write"
    # Existing behaviour intact.
    assert _memory_kind("memory.get", None) == "read"
    assert _memory_kind("memory.upsert", None) == "write"
    assert _memory_kind("tool.call", None) is None


# ── pulse/causal_skeleton: docs match behaviour (no phantom class) ─────────


def test_causal_skeleton_docs_no_phantom_confounding_class():
    from vfairness.operations.pulse import causal_skeleton as cs

    doc = cs.__doc__ or ""
    src = inspect.getsource(cs)
    # The module never emits a 'confounding' path/edge kind, so the
    # docstring must not promise one as a produced classification.
    assert "confounding : a feature" not in doc
    assert '"kind": "confounding"' not in src
    assert "'kind': 'confounding'" not in src
    # The direct-path floor is named for what it gates.
    assert hasattr(cs, "_DIRECT_MIN")
    assert not hasattr(cs, "_CONFOUND_MIN")


def test_causal_skeleton_direct_path_still_renders():
    """Behaviour unchanged by the rename: a protected attribute associated
    with the outcome yields a direct path."""
    from vfairness.operations.pulse.causal_skeleton import build_causal_skeleton

    rng = np.random.default_rng(11)
    n = 300
    g = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "grp": np.where(g == 1, "a", "b"),
            "out": (rng.random(n) < (0.3 + 0.4 * g)).astype(int),
        }
    )
    sk = build_causal_skeleton(df, ["grp"], "out")
    kinds = {p["kind"] for p in sk.paths}
    assert "direct" in kinds
    assert "confounding" not in kinds


# ── cicd/gate: threshold 0.0 is enforced and must display ─────────────────


def test_gate_zero_threshold_not_rendered_as_na():
    from vfairness.operations.cicd.gate import GateDecision, GateStatus, MetricEvaluation

    ev0 = MetricEvaluation(
        metric_name="spd", value=0.1, threshold=0.0, passed=False, is_blocking=True, message="x"
    )
    ev_none = MetricEvaluation(
        metric_name="unbounded",
        value=0.1,
        threshold=None,
        passed=True,
        is_blocking=False,
        message="y",
    )
    gd = GateDecision(approved=False, status=GateStatus.BLOCKED, metric_evaluations=[ev0, ev_none])
    report = gd.to_markdown_report()
    assert "0.0000" in report  # the enforced zero threshold
    assert report.count("N/A") == 1  # only the truly absent threshold


# ── cicd/validator: JUnit XML escapes special characters ──────────────────


def test_junit_xml_escapes_special_characters():
    from vfairness.operations.cicd.validator import (
        DataValidationResult,
        ValidationIssue,
        ValidationSeverity,
    )

    issue = ValidationIssue(
        issue_type='bad<type>&"q"',
        severity=ValidationSeverity.ERROR,
        message='msg with <angle> & "quotes"',
        details={"k": "<v>"},
    )
    res = DataValidationResult(passed=False, issues=[issue], metrics={}, summary="s")
    root = ET.fromstring(res.to_junit_xml())  # must parse
    case = root.find("testcase")
    assert case.get("name") == 'bad<type>&"q"'
    failure = case.find("failure")
    assert failure.get("message") == 'msg with <angle> & "quotes"'
    assert "<v>" in failure.text


# ── monitoring/alerts: explicit 0.0 drift values are honored ──────────────


def test_alert_explicit_zero_drift_values_honored():
    from vfairness.operations.monitoring.alerts import FairnessAlertPrioritizer

    am = FairnessAlertPrioritizer()
    pay = am.create_alert({"drift_score": 0.7, "mean_shift": 0.3}, drift_score=0.0, mean_shift=0.0)
    assert pay.drift_score == 0.0
    assert pay.mean_shift == 0.0
    # Omitted arguments still fall back to the event values.
    pay2 = am.create_alert({"drift_score": 0.7, "mean_shift": 0.3})
    assert pay2.drift_score == 0.7
    assert pay2.mean_shift == 0.3
    # Missing everywhere still never crashes, but it is NOT 0.0.
    #
    # BGL S2, 2026-09-16. This block asserted `pay3.drift_score == 0.0` under
    # the comment "degrades to 0.0", which is the fabrication: 0.000 is the
    # reading for a metric that did not move, and it was being printed into
    # the payload, to_dict() and the alert message for a metric nobody read,
    # silently, on alerts up to CRITICAL. The subject of the assertion (a
    # fully empty event is handled and does not raise) is kept; the value
    # moves to the third state and now says so.
    with pytest.warns(UserWarning, match="COULD NOT CHECK"):
        pay3 = am.create_alert({})
    assert math.isnan(pay3.drift_score)
    assert math.isnan(pay3.mean_shift)


# ── monitoring/tracker: seasonal slots correct for period != 7 ─────────────


def _analyzer_with(dates, values):
    from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

    tr = TemporalFairnessAnalyzer()
    tr._daily_metrics = pd.DataFrame({"date": pd.to_datetime(list(dates)), "m": list(values)})
    return tr


def test_seasonal_pattern_nonweekly_period_true_cycle():
    dates = pd.date_range("2026-01-05", periods=9, freq="D")
    vals = [float((d.toordinal() - 1) % 3) for d in dates]
    tr = _analyzer_with(dates, vals)
    pat = tr.detect_seasonal_pattern("m", period=3)
    # A true 3-day cycle groups equal values into their own slots.
    assert pat == {0: 0.0, 1: 1.0, 2: 2.0}


def test_seasonal_pattern_period7_identical_to_weekday():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2026-01-05", periods=21, freq="D")
    vals = rng.random(21)
    tr = _analyzer_with(dates, vals)
    pat = tr.detect_seasonal_pattern("m", period=7)
    df = tr._daily_metrics.copy()
    df["wd"] = df["date"].dt.dayofweek
    expected = df.groupby("wd")["m"].mean().to_dict()
    assert pat == expected
    # And slot 0 is Monday (2026-01-05 is a Monday).
    assert pat[0] == pytest.approx(np.mean([vals[0], vals[7], vals[14]]))


# ── causal/refute: crashed refuters excluded, not counted as failures ─────


class _FakeEstimate:
    value = 1.0


class _FakeRefutation:
    def __init__(self, new_effect):
        self.new_effect = new_effect
        self.refutation_result = {"p_value": 0.5}


def _install_fake_dowhy(monkeypatch, crash_methods):
    class _FakeModel:
        def __init__(self, **kw):
            pass

        def identify_effect(self, **kw):
            return "estimand"

        def estimate_effect(self, *a, **kw):
            return _FakeEstimate()

        def refute_estimate(self, estimand, base, method_name=None):
            if method_name in crash_methods:
                raise RuntimeError("refuter unavailable")
            return _FakeRefutation(
                0.0
                if method_name in ("placebo_treatment_refuter", "dummy_outcome_refuter")
                else 1.0
            )

    fake = types.ModuleType("dowhy")
    fake.CausalModel = _FakeModel
    monkeypatch.setitem(sys.modules, "dowhy", fake)


def test_refute_crashed_refuters_do_not_penalize_score(monkeypatch):
    _install_fake_dowhy(monkeypatch, crash_methods={"data_subset_refuter", "dummy_outcome_refuter"})
    from vfairness.operations.causal.refute import run_refutation_suite

    rr = run_refutation_suite(
        "graph []", pd.DataFrame({"t": [0, 1, 0, 1, 0, 1], "y": [0, 1, 0, 1, 1, 0]}), "t", "y"
    )
    # Both refuters that ran passed: 2/2, not 2/4.
    assert rr.robustness_score == 1.0
    assert any("excluded" in w for w in rr.warnings)
    crashed = [o for o in rr.outcomes if any("Unavailable" in n for n in o.notes)]
    assert len(crashed) == 2
    assert all(o.refuted_effect is None for o in crashed)


def test_refute_all_crashed_yields_no_score(monkeypatch):
    _install_fake_dowhy(
        monkeypatch,
        crash_methods={
            "placebo_treatment_refuter",
            "random_common_cause",
            "data_subset_refuter",
            "dummy_outcome_refuter",
        },
    )
    from vfairness.operations.causal.refute import run_refutation_suite

    rr = run_refutation_suite(
        "graph []", pd.DataFrame({"t": [0, 1, 0, 1, 0, 1], "y": [0, 1, 0, 1, 1, 0]}), "t", "y"
    )
    assert rr.robustness_score is None
    assert any("No refuter could run" in w for w in rr.warnings)


def test_refute_genuine_failure_still_penalizes(monkeypatch):
    """A refuter that RUNS and refutes must still count as a failure."""
    _install_fake_dowhy(monkeypatch, crash_methods=set())

    # random_common_cause / data_subset return 1.0 (pass: near observed);
    # placebo / dummy return 0.0 (pass: near zero). Flip one to fail.
    import vfairness.operations.causal.refute as rf

    real_passed = rf._passed

    def _flip(test_name, observed, refuted):
        if test_name == "Random common cause":
            return False
        return real_passed(test_name, observed, refuted)

    monkeypatch.setattr(rf, "_passed", _flip)
    rr = rf.run_refutation_suite(
        "graph []", pd.DataFrame({"t": [0, 1, 0, 1, 0, 1], "y": [0, 1, 0, 1, 1, 0]}), "t", "y"
    )
    assert rr.robustness_score == pytest.approx(3.0 / 4.0)


# ── pulse/llm_probe: refusal findings drive per-axis isSignificant ─────────


_REFUSE_TEXT = "I cannot assist with this request."
_ANSWER_TEXT = "Here is the requested assessment of the profile for the file."


def _serve(decide):
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
                    ]
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


def test_refusal_only_disparity_sets_axis_significant(monkeypatch):
    """Endpoint refuses two descriptor arms but answers everyone else with
    identical text. The sentiment channel is forced flat (constant scores,
    independent of which sentiment scorer is installed) so ONLY the
    refusal channel can fire; the per-axis isSignificant flag must then
    reflect the family-wise-significant refusal findings instead of
    contradicting them."""
    import vfairness.operations.pulse.llm_probe as lp

    monkeypatch.setattr(
        lp,
        "_sentiment_scores",
        lambda responses: [0.0 for r in (responses or []) if isinstance(r, str) and r.strip()],
    )

    def decide(body):
        if "61-year-old" in body or "who is Muslim" in body:
            return _REFUSE_TEXT
        return _ANSWER_TEXT

    srv = _serve(decide)
    try:
        port = srv.server_address[1]
        out = lp.llm_probe_pulse(
            {
                "endpoint_url": "http://127.0.0.1:{0}/v1/chat/completions".format(port),
                "api_format": "openai",
                "model_name": "stub",
                "n_runs": 2,
                # The stub is on loopback; the egress guard refuses 127.0.0.1 unless
                # a caller opts in explicitly. Production callers must NOT set this.
                "allow_loopback": True,
            },
            protected_groups={"name": ["Greg", "Jamal"]},
        )
    finally:
        srv.shutdown()

    d = out["data"]
    rows = {r["axis"]: r for r in d["generative"]["perAxis"]}
    refusal_axes = sorted(
        f["attribute"] for f in d["bias"] if f["type"] == "llm_refusal_rate_disparity"
    )
    assert refusal_axes == ["age", "religion"]
    for axis in ("age", "religion"):
        row = rows[axis]
        assert row["replicatedIn"] == 0  # sentiment stayed quiet
        assert row["isSignificant"] is True  # refusal finding drives it
    # Axes with no refusal disparity and no sentiment signal stay quiet.
    assert rows["name"]["isSignificant"] is False
    assert rows["disability"]["isSignificant"] is False
