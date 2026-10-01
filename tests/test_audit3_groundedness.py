"""Third-iteration audit pins for the groundedness (validity) unit.

Each test drives the exact scenario that used to give the wrong answer and asserts
it now gives the right one (negative/defect case first), plus a does-not-overcorrect
control so a genuine measurement still passes. No real network is used: SSRF cases
use IP literals, which ``validate_endpoint`` checks directly without DNS resolution.
"""

import warnings

import pytest

from vfairness.operations.validity.task_handlers import _build_scorer
from vfairness.validity.aggregate import aggregate_validity
from vfairness.validity.groundedness import (
    GroundednessResult,
    GroundednessScorer,
    ValidityScorerUnavailableWarning,
)
from vfairness.validity.judge import LlmGroundednessJudge


class _Rung:
    """A stand-in rung (sidecar or judge) returning a fixed verdict dict."""

    def __init__(self, verdict):
        self._verdict = verdict

    def judge_groundedness(self, *, answer, contexts, question, language):
        return dict(self._verdict)


# ---------------------------------------------------------------------------
# HIGH: the central guard must fail CLOSED on a boolean score (True -> 1.0).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rung_kw", ["sidecar", "judge"])
def test_central_guard_refuses_boolean_true_score(rung_kw):
    # DEFECT: float(True) == 1.0, so {"groundedness": True} clamped to a perfect 1.0,
    # available=True (a garbage reply sealing a PASS) at BOTH call sites, not only the
    # LLM judge. The bool defense in judge.py never covered the central guard.
    rung = _Rung({"groundedness": True, "unsupported_spans": ["claim X unsupported"]})
    scorer = GroundednessScorer(**{rung_kw: rung})
    with pytest.warns(ValidityScorerUnavailableWarning):
        r = scorer.score("The sky is green.", ["The sky is blue."])
    assert r.available is False
    assert r.value is None
    assert r.hallucination_rate is None
    assert "boolean" in r.note.lower()


@pytest.mark.parametrize("rung_kw", ["sidecar", "judge"])
def test_central_guard_refuses_boolean_false_score(rung_kw):
    # float(False) == 0.0 would read as a real, available "totally ungrounded" score;
    # a boolean is not a measurement and must be refused, not scored 0.0-available.
    scorer = GroundednessScorer(**{rung_kw: _Rung({"groundedness": False})})
    with pytest.warns(ValidityScorerUnavailableWarning):
        r = scorer.score("a", ["c"])
    assert r.available is False
    assert r.value is None


@pytest.mark.parametrize("value", [0.0, 0.5, 0.9, 1.0])
def test_central_guard_accepts_genuine_float_scores(value):
    # DOES-NOT-OVERCORRECT: a real float (including a genuine 0.0 = fully ungrounded and
    # a genuine 1.0 = fully grounded) is still a measurement and must pass through.
    scorer = GroundednessScorer(sidecar=_Rung({"groundedness": value}))
    r = scorer.score("a", ["c"])
    assert r.available is True
    assert r.value == pytest.approx(value)
    assert r.hallucination_rate == pytest.approx(1.0 - value)


# ---------------------------------------------------------------------------
# LOW: a finite out-of-range HIGH score must be REFUSED, not clamped UP to 1.0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [47.0, 1.4, -5.0])
def test_central_guard_refuses_out_of_range_score(bad):
    # DEFECT: 47.0 / 1.4 clamped UP to 1.0 (favorable fail-open); the guard now refuses
    # any value outside [0,1] (either side), matching the NaN/Infinity treatment.
    scorer = GroundednessScorer(sidecar=_Rung({"groundedness": bad}))
    with pytest.warns(ValidityScorerUnavailableWarning):
        r = scorer.score("a", ["c"])
    assert r.available is False
    assert r.value is None
    assert "out-of-range" in r.note.lower()


@pytest.mark.parametrize("bad", [47.0, 1.4, -5.0])
def test_judge_refuses_out_of_range_score(bad, monkeypatch):
    j = LlmGroundednessJudge("http://127.0.0.1:11435/v1/chat/completions", "m")
    monkeypatch.setattr(
        j, "_complete", lambda *a: f'{{"groundedness": {bad}, "unsupported_claims": []}}'
    )
    out = j.judge_groundedness(answer="a", contexts=["c"], question=None, language="en")
    assert out["groundedness"] is None
    assert "out-of-range" in out["note"].lower()


def test_out_of_range_guard_absorbs_tiny_float_overshoot():
    # DOES-NOT-OVERCORRECT: a value a hair over 1.0 from float arithmetic is clamped,
    # not refused (the refusal only targets clearly-invalid values, > 1.0 + 1e-6).
    scorer = GroundednessScorer(sidecar=_Rung({"groundedness": 1.0 + 1e-9}))
    r = scorer.score("a", ["c"])
    assert r.available is True
    assert r.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MEDIUM: VG-005 must be the SHARE OF ANSWERS with unsupported content
# (answer incidence), not the claim-weighted 1 - mean(groundedness).
# ---------------------------------------------------------------------------


def test_vg005_is_answer_incidence_all_hallucinated_batch_fails_gate():
    # DEFECT: ten answers each 90% grounded => claim-weighted mean = 0.9 => VG-005 = 0.1
    # PASSED the 0.1 gate, yet EVERY answer contains an unsupported claim. VG-005 must be
    # 1.0 (100% of answers hallucinated) and FAIL the gate.
    rs = [GroundednessResult(value=0.9, available=True) for _ in range(10)]
    agg = aggregate_validity(rs)
    thr = agg["VG-005"]["threshold"]
    assert agg["VG-005"]["mean"] == pytest.approx(1.0)
    assert not (agg["VG-005"]["mean"] <= thr)  # gate FAILS, as it must
    # VG-001 (mean groundedness) is unchanged by the VG-005 fix.
    assert agg["VG-001"]["mean"] == pytest.approx(0.9)


def test_vg005_answer_incidence_mixed_batch():
    # 3 perfectly-grounded answers + 1 with an unsupported claim => share = 1/4 = 0.25.
    rs = [
        GroundednessResult(value=1.0, available=True),
        GroundednessResult(value=1.0, available=True),
        GroundednessResult(value=1.0, available=True),
        GroundednessResult(value=0.9, available=True),
    ]
    agg = aggregate_validity(rs)
    assert agg["VG-005"]["mean"] == pytest.approx(0.25)


def test_vg005_counts_perfect_value_with_unsupported_spans():
    # A value of 1.0 that nonetheless carries unsupported spans still counts as an
    # answer with unsupported content (defensive: the span evidence, not just the number).
    rs = [
        GroundednessResult(value=1.0, unsupported_spans=["leftover claim"], available=True),
        GroundednessResult(value=1.0, available=True),
    ]
    agg = aggregate_validity(rs)
    assert agg["VG-005"]["mean"] == pytest.approx(0.5)


def test_vg005_clean_batch_is_zero_not_overcorrected():
    # DOES-NOT-OVERCORRECT: a genuinely clean batch (all perfectly grounded, no spans)
    # must report VG-005 = 0.0 and PASS the gate.
    rs = [GroundednessResult(value=1.0, available=True) for _ in range(5)]
    agg = aggregate_validity(rs)
    assert agg["VG-005"]["mean"] == pytest.approx(0.0)
    assert agg["VG-005"]["mean"] <= agg["VG-005"]["threshold"]


# ---------------------------------------------------------------------------
# MEDIUM: the payload-supplied judge endpoint must go through the SSRF guard,
# not just a scheme check. No real network (IP literals only).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",  # cloud metadata (link-local)
        "http://10.0.0.5:11435/v1/chat/completions",  # RFC1918 internal service
        "http://192.168.1.10:11435/v1/chat/completions",  # RFC1918 internal service
    ],
)
def test_build_scorer_refuses_ssrf_endpoint(url, monkeypatch):
    # DEFECT: only an http(s) scheme check gated the payload endpoint, so a metadata /
    # RFC1918 target was accepted and the judge would POST to it and parse the reply as
    # the sealed verdict. It must now be refused -> scorer fail-closed (no rung).
    monkeypatch.delenv("VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE", raising=False)
    scorer = _build_scorer({"judge": {"endpoint_url": url, "model_name": "m"}})
    assert scorer.available is False


def test_build_scorer_accepts_loopback_operator_endpoint(monkeypatch):
    # DOES-NOT-OVERCORRECT: a local judge (Ollama on loopback) is the first-class
    # operator case and must still build a working scorer.
    monkeypatch.delenv("VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE", raising=False)
    scorer = _build_scorer(
        {"judge": {"endpoint_url": "http://127.0.0.1:11435/v1/chat/completions", "model_name": "m"}}
    )
    assert scorer.available is True


def test_build_scorer_opt_in_allows_private_endpoint(monkeypatch):
    # An explicit operator opt-in widens the guard to a private LAN judge (matches the
    # deliberate allow_loopback posture LLMApiProxy exposes).
    monkeypatch.setenv("VFAIRNESS_VALIDITY_JUDGE_ALLOW_PRIVATE", "1")
    scorer = _build_scorer(
        {"judge": {"endpoint_url": "http://10.0.0.5:11435/v1/chat/completions", "model_name": "m"}}
    )
    assert scorer.available is True


def test_build_scorer_no_endpoint_stays_fail_closed(monkeypatch):
    # Sanity: with no endpoint at all the guard is not reached and the scorer refuses.
    monkeypatch.delenv("VFAIRNESS_VALIDITY_JUDGE_URL", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer = _build_scorer({})
    assert scorer.available is False
