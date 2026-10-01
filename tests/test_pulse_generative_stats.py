"""Deterministic tests for Pulse close-plan Phase 3 (G-06, G-11, G-12).

G-06: the generative battery at the tabular bar; all group pairs per
attribute, run-wide family-wise BH-FDR across every (attribute, pair,
metric) p-value, per-attribute minimum-detectable-effect power
disclosure, and a text-specific dataQuality section.

G-11: regression + ranking routing; continuous and rank model outputs
are never silently median-thresholded into a fake binary decision but
routed to the library's regression-parity / ranking-exposure batteries
with the same perVariable row shape.

G-12: the group-calibration stage; per-group Expected Calibration Error
with between-group gaps feeding calibration_disparity findings.

Every test runs locally: deterministic fixtures, no network, no external services.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse

# ── fixtures ────────────────────────────────────────────────────

_POS_TEXT = (
    "The candidate is excellent and we recommend hiring with strong "
    "confidence in the assessment of this application."
)
_NEG_TEXT = (
    "The candidate is weak and we do not recommend hiring due to the "
    "poor results in the assessment of this application."
)


def _three_group_generative_df() -> pd.DataFrame:
    """Groups A (30) and B (25) carry the SAME text, so (A, B) is a true
    null pair; C (20) carries a starkly different text, so (A, C) and
    (B, C) hold the planted disparity. Distinct counts keep the
    value_counts ordering (A, B, C) deterministic."""
    rows = [{"prompt": "p", "response": _POS_TEXT, "group": "A"} for _ in range(30)]
    rows += [{"prompt": "p", "response": _POS_TEXT, "group": "B"} for _ in range(25)]
    rows += [{"prompt": "p", "response": _NEG_TEXT, "group": "C"} for _ in range(20)]
    return pd.DataFrame(rows)


def _continuous_df(n_per: int = 300) -> pd.DataFrame:
    """Score-only artefact (no binary decision column) with a planted
    mean shift: male scores center on 0.65, female on 0.45."""
    rng = np.random.default_rng(11)
    male = np.clip(rng.normal(0.65, 0.10, n_per), 0.0, 1.0)
    female = np.clip(rng.normal(0.45, 0.10, n_per), 0.0, 1.0)
    return pd.DataFrame(
        {
            "gender": ["male"] * n_per + ["female"] * n_per,
            "score": np.concatenate([male, female]),
            "experience_years": rng.integers(0, 30, 2 * n_per),
        }
    )


def _rank_df(n: int = 200) -> pd.DataFrame:
    """Full ranking 1..n where grpA holds the top 120 slots and grpB is
    pushed to the bottom 80: a planted exposure disparity."""
    rng = np.random.default_rng(5)
    n_a, n_b = 120, 80
    group = np.array(["grpA"] * n_a + ["grpB"] * n_b)
    ranks = np.concatenate(
        [
            rng.permutation(np.arange(1, n_a + 1)),
            rng.permutation(np.arange(n_a + 1, n + 1)),
        ]
    )
    return pd.DataFrame(
        {
            "group": group,
            "rank": ranks,
            "feature": rng.normal(size=n),
        }
    )


def _calibration_df(n_per: int = 300) -> pd.DataFrame:
    """Binary decisions plus a probability score and ground truth.
    Team A is well calibrated (label ~ Bernoulli(score)); team B is
    deliberately miscalibrated (scores 0.80-0.95, base rate 0.20)."""
    rng = np.random.default_rng(3)
    score_a = rng.uniform(0.05, 0.95, n_per)
    label_a = (rng.uniform(size=n_per) < score_a).astype(int)
    score_b = rng.uniform(0.80, 0.95, n_per)
    label_b = (rng.uniform(size=n_per) < 0.20).astype(int)
    score = np.concatenate([score_a, score_b])
    label = np.concatenate([label_a, label_b])
    return pd.DataFrame(
        {
            "team": ["A"] * n_per + ["B"] * n_per,
            "score": score,
            "label": label,
            "prediction": (score >= 0.5).astype(int),
        }
    )


@pytest.fixture(scope="module")
def gen_result():
    r = run_pulse(
        _three_group_generative_df(),
        {"source_kind": "prompts_outputs", "protected_attributes": ["group"]},
    )
    assert r["success"] is True
    return r["data"]


def _comparison(entry, a, b):
    for c in entry["comparisons"]:
        if {c["groupA"], c["groupB"]} == {a, b}:
            return c
    raise AssertionError(f"missing comparison {a} vs {b}")


# ── G-06: generative battery ────────────────────────────────────


def test_generative_all_pairs_compared(gen_result):
    assert gen_result["sourceKind"] == "prompts_outputs"
    per_attr = gen_result["generative"]["perAttribute"]
    assert len(per_attr) == 1
    entry = per_attr[0]
    assert entry["attribute"] == "group"
    assert entry["mode"] == "all_pairs"
    assert len(entry["comparisons"]) == 3, "expected (A,B), (A,C), (B,C)"
    seen = {frozenset((c["groupA"], c["groupB"])) for c in entry["comparisons"]}
    assert seen == {frozenset(("A", "B")), frozenset(("A", "C")), frozenset(("B", "C"))}


def test_generative_familywise_fdr_significance(gen_result):
    entry = gen_result["generative"]["perAttribute"][0]
    planted = _comparison(entry, "A", "C")
    null = _comparison(entry, "A", "B")
    sig = [m for m in planted["metrics"] if m.get("is_significant")]
    assert sig, "the planted A-vs-C disparity must survive family-wise FDR"
    for m in sig:
        assert m.get("p_value_family_adjusted") is not None
        assert m["p_value_family_adjusted"] <= 0.05
    assert not any(m.get("is_significant") for m in null["metrics"]), (
        "the null A-vs-B pair must NOT be significant after family-wise FDR"
    )
    # Every scored metric carries the family-adjusted p-value.
    assert all("p_value_family_adjusted" in m for m in planted["metrics"])
    # Findings were raised off the family-wise decisions.
    assert any(f["type"] == "generative_output_disparity" for f in gen_result["bias"])
    assert gen_result["verdict"]["tone"] in ("warn", "critical")


def test_generative_data_quality_present(gen_result):
    dq = gen_result["dataQuality"]
    assert dq["rows"] == 75
    assert dq["emptyTexts"] == 0
    assert dq["duplicateTexts"] > 0, "repeated fixture texts are duplicates"
    assert dq["perGroupCounts"]["group"] == {"A": 30, "B": 25, "C": 20}
    assert dq["tone"] in ("pass", "warn")
    assert isinstance(dq["checks"], list) and dq["checks"]


def test_generative_min_detectable_effect_disclosed(gen_result):
    entry = gen_result["generative"]["perAttribute"][0]
    mde = entry["minDetectableEffect"]
    assert mde["nSmallestGroup"] == 20
    assert abs(mde["d"] - 2.8 * (2.0 / 20.0) ** 0.5) < 1e-9
    assert "weak evidence" in mde["plain"]
    # Run-wide power note on the scope block names the smallest n.
    assert "n=20" in gen_result["scope"]["powerNote"]


# ── G-11: continuous (regression) routing ───────────────────────


def test_continuous_output_routes_to_regression_battery():
    r = run_pulse(_continuous_df(), {"protected_attributes": ["gender"]})
    assert r["success"] is True, r.get("error")
    assert "error" not in r
    d = r["data"]
    assert d["outputType"] == "continuous"
    assert d["dataPreparation"]["outputType"] == "continuous"
    row = next(v for v in d["perVariable"] if v["attribute"] == "gender")
    assert row["assessable"] is True
    assert row["worstGroup"] == "female"
    assert row["gap"] > 0.10, "planted 0.2 raw mean shift must surface"
    assert row["significant"] is True
    assert row["fourFifthsRatio"] is None
    assert row["outputType"] == "continuous"
    assert row["gapUnit"] == "normalized_mean_difference"
    reg = row.get("regression")
    assert reg and reg["metric"] == "mean_prediction_difference"
    assert reg["value"] > 0.0
    assert "No prediction column" not in str(d.get("verdict", {}))
    assert d["verdict"]["tone"] in ("warn", "critical")


# ── G-11: rank routing ──────────────────────────────────────────


def test_rank_output_routes_to_ranking_battery():
    r = run_pulse(_rank_df(), {"protected_attributes": ["group"]})
    assert r["success"] is True, r.get("error")
    d = r["data"]
    assert d["outputType"] == "rank"
    row = next(v for v in d["perVariable"] if v["attribute"] == "group")
    assert row["assessable"] is True
    assert row["worstGroup"] == "grpB", "bottom-ranked group must be worst"
    assert row["gap"] > 0.0
    assert row["significant"] is True
    assert row["fourFifthsRatio"] is None
    assert row["outputType"] == "rank"
    rk = row.get("ranking")
    assert rk is not None, "library ranking battery must populate"
    assert rk["exposureParityDifference"] > 0.0
    assert rk["ndkl"] >= 0.0


# ── G-12: calibration stage ─────────────────────────────────────


def test_calibration_stage_and_finding():
    r = run_pulse(_calibration_df(), {"protected_attributes": ["team"]})
    assert r["success"] is True, r.get("error")
    d = r["data"]
    cal = d["calibration"]
    assert cal["available"] is True
    pa = next(x for x in cal["perAttribute"] if x["attribute"] == "team")
    assert pa["mostMiscalibratedGroup"] == "B"
    assert pa["maxGap"] >= 0.10
    assert pa["tone"] == "critical"
    worst = max(pa["groups"], key=lambda g: g["ece"])
    assert worst["group"] == "B", "the miscalibrated group carries the gap"
    assert worst["n"] == 300
    assert any(f["type"] == "calibration_disparity" for f in d["bias"]), (
        "significant calibration gaps must reach the bias findings"
    )
