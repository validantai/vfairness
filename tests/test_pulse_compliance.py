"""Phase-4 compliance-packaging tests for the Pulse orchestrator.

Covers the close-plan engine items:
  G-13 legal admissibility, G-18 LL144 + Art. 10 exports, G-19
  harm-direction routing, G-24 ISO/IEC TR 24027 crosswalk, G-31
  historical context, G-32 reference-group override + targets, G-34
  recheck window with a secret-free suite echo.

Every test is deterministic and local: seeded fixtures, no network,
no external services.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.regulatory import (
    _add_months,
    build_recheck,
    build_targets,
)

# ── fixtures ────────────────────────────────────────────────────────────────

N_UNKNOWN = 12  # rows with a missing protected value in the hiring fixture


def _hiring_us_df(n: int = 400) -> pd.DataFrame:
    """Stark gap: males selected at ~0.7, females at ~0.2, plus a
    continuous score and N_UNKNOWN rows with missing gender."""
    rng = np.random.default_rng(11)
    half = n // 2
    gender = np.array(["male"] * half + ["female"] * half, dtype=object)
    pred = np.concatenate([rng.binomial(1, 0.7, half), rng.binomial(1, 0.2, half)])
    score = pred * 0.5 + rng.uniform(0.0, 0.5, n)
    df = pd.DataFrame(
        {
            "gender": gender,
            "prediction": pred,
            "score": score,
            "feature": rng.normal(size=n),
        }
    )
    df.loc[df.index[:N_UNKNOWN], "gender"] = None
    return df


def _credit_eu_df(n: int = 400) -> pd.DataFrame:
    """EU lending fixture WITH ground truth; base rates differ across
    groups (~0.7 vs ~0.4) so the impossibility acknowledgment fires."""
    rng = np.random.default_rng(23)
    half = n // 2
    group = np.array(["g1"] * half + ["g2"] * half, dtype=object)
    label = np.concatenate([rng.binomial(1, 0.7, half), rng.binomial(1, 0.4, half)])
    noise = rng.random(n) < 0.85
    pred = np.where(noise, label, 1 - label)
    score = pred * 0.4 + label * 0.2 + rng.uniform(0.0, 0.4, n)
    return pd.DataFrame(
        {
            "group": group,
            "label": label,
            "prediction": pred,
            "score": score,
            "feature": rng.normal(size=n),
        }
    )


def _three_group_df(n_a: int = 150, n_b: int = 90, n_c: int = 60) -> pd.DataFrame:
    """Three groups: A (majority, ~0.7), B (~0.5), C (worst, ~0.2)."""
    rng = np.random.default_rng(5)
    group = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c, dtype=object)
    pred = np.concatenate(
        [
            rng.binomial(1, 0.7, n_a),
            rng.binomial(1, 0.5, n_b),
            rng.binomial(1, 0.2, n_c),
        ]
    )
    return pd.DataFrame(
        {
            "group": group,
            "prediction": pred,
            "feature": rng.normal(size=len(group)),
        }
    )


HIRING_INPUTS = {
    "domain": "hiring",
    "jurisdiction": "US",
    "protected_attributes": ["gender"],
    "harm_direction": "punitive",
    "scores_exposed": True,
    "target_ratio": 0.9,
    # planted secrets: MUST NOT survive into recheck.suite
    "auth_token": "sekrit-token-123",
    "model_base64": "QUJDREVG",
    "model_envelope": {"cipher": "aes", "blob": "zzz"},
    "org_api_key": "org-sekrit-456",
}


@pytest.fixture(scope="module")
def hiring_us():
    r = run_pulse(_hiring_us_df(), dict(HIRING_INPUTS))
    assert r["success"] is True
    return r["data"]


@pytest.fixture(scope="module")
def credit_eu():
    r = run_pulse(
        _credit_eu_df(),
        {
            "domain": "credit scoring",
            "jurisdiction": "EU",
            "protected_attributes": ["group"],
            "harm_direction": "assistive",
            "scores_exposed": True,
            "artifact_label": "loans-2026Q1.csv",
            "artifact_hash": "abc123def",
        },
    )
    assert r["success"] is True
    return r["data"]


@pytest.fixture(scope="module")
def unknown_domain():
    """Also the no-override baseline for the reference-group tests."""
    r = run_pulse(
        _three_group_df(),
        {
            "domain": "asteroid mining",
            "jurisdiction": "",
            "protected_attributes": ["group"],
        },
    )
    assert r["success"] is True
    return r["data"]


@pytest.fixture(scope="module")
def ref_override_b():
    r = run_pulse(
        _three_group_df(),
        {
            "domain": "generic",
            "jurisdiction": "",
            "protected_attributes": ["group"],
            "reference_group": {"group": "B"},
        },
    )
    assert r["success"] is True
    return r["data"]


@pytest.fixture(scope="module")
def ref_override_missing():
    r = run_pulse(
        _three_group_df(),
        {
            "domain": "generic",
            "jurisdiction": "",
            "protected_attributes": ["group"],
            "reference_group": {"group": "Z"},
        },
    )
    assert r["success"] is True
    return r["data"]


def _pv_row(data, attribute):
    row = next((v for v in data["perVariable"] if v.get("attribute") == attribute), None)
    assert row is not None, f"no perVariable row for {attribute}"
    return row


# ── G-18: LL144 export (hiring + US) ────────────────────────────────────────


def test_ll144_applicable_on_hiring_us(hiring_us):
    ll = hiring_us["regulatoryExports"]["ll144"]
    assert ll["applicable"] is True
    assert ll["impactRatios"], "impact ratios must be computed"
    by_group = {r["group"]: r for r in ll["impactRatios"] if r["attribute"] == "gender"}
    assert "male" in by_group and "female" in by_group
    assert by_group["male"]["impactRatio"] == pytest.approx(1.0)
    assert by_group["female"]["impactRatio"] < 0.5
    for g in ("male", "female"):
        assert by_group[g]["n"] > 0
        assert 0.0 <= by_group[g]["selectionRate"] <= 1.0
        # A continuous score exists in this fixture, so the above-median
        # scoring rate must be a real number, not null.
        assert isinstance(by_group[g]["aboveMedianRate"], float)


def test_ll144_counts_unknown_demographics(hiring_us):
    ll = hiring_us["regulatoryExports"]["ll144"]
    assert ll["unknownDemographics"] == N_UNKNOWN


def test_ll144_self_serve_note(hiring_us):
    notes = hiring_us["regulatoryExports"]["ll144"]["notes"]
    assert any("not an independent LL144 bias audit" in n for n in notes)


def test_ll144_not_applicable_outside_us_employment(credit_eu, unknown_domain):
    for data in (credit_eu, unknown_domain):
        ll = data["regulatoryExports"]["ll144"]
        assert ll["applicable"] is False
        assert ll["reason"]


# ── G-18: Art. 10 export (EU) ───────────────────────────────────────────────


def test_art10_applicable_on_eu_with_real_run_data(credit_eu):
    art = credit_eu["regulatoryExports"]["art10"]
    assert art["applicable"] is True
    sections = art["sections"]
    assert len(sections) == 4
    ids = [s["id"] for s in sections]
    assert ids == ["data_sources", "data_governance", "bias_examination", "bias_mitigation"]
    for s in sections:
        assert s["title"] and s["body"]
    src = sections[0]["body"]
    assert "loans-2026Q1.csv" in src and "abc123def" in src
    # The bias-examination section must carry the run's actual verdict.
    assert credit_eu["verdict"]["headline"].split('"')[0].strip()[:20] in sections[2]["body"]


def test_art10_not_applicable_outside_eu(hiring_us):
    art = hiring_us["regulatoryExports"]["art10"]
    assert art["applicable"] is False
    assert art["sections"] == []


# ── G-24: ISO/IEC TR 24027 crosswalk ────────────────────────────────────────


def test_iso24027_maps_computed_metrics_only(hiring_us, credit_eu):
    ours_no_truth = {m["ourMetric"] for m in hiring_us["regulatoryExports"]["iso24027"]}
    assert "selection_rate_difference" in ours_no_truth
    assert "four_fifths_ratio" in ours_no_truth
    assert "cramers_v_association" in ours_no_truth
    # No ground truth in the hiring fixture: equalized odds was NOT
    # computed, so it must not be claimed.
    assert "equalized_odds_difference" not in ours_no_truth

    ours_truth = {m["ourMetric"] for m in credit_eu["regulatoryExports"]["iso24027"]}
    assert "equalized_odds_difference" in ours_truth
    assert "equal_opportunity_difference" in ours_truth
    for m in hiring_us["regulatoryExports"]["iso24027"]:
        assert "ISO/IEC TR 24027" in m["clause"]
        assert m["isoName"]


def test_legal_context_as_of_is_server_date(hiring_us):
    as_of = hiring_us["regulatoryExports"]["legalContextAsOf"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", as_of)


# ── G-13: legal admissibility ───────────────────────────────────────────────


def test_legal_admissibility_present_with_coverage(hiring_us):
    la = hiring_us["legalAdmissibility"]
    assert la["coverage"] in ("covered", "partial", "uncovered")
    # hiring + US maps to the recruitment.us-federal rule pack.
    assert la["coverage"] == "covered"
    assert la["useCase"] == "recruitment"
    assert la["jurisdictionId"] == "us-federal"
    assert isinstance(la["findings"], list)
    assert "sourceRevision" in la
    # The gender column must be classified (monitoring_only under
    # Title VII EEO monitoring in the curated pack).
    gender = [f for f in la["findings"] if f["column"] == "gender"]
    assert gender, "gender column must yield an admissibility finding"


# ── G-31: historical context ────────────────────────────────────────────────


def test_historical_context_hiring(hiring_us):
    hc = hiring_us["historicalContext"]
    assert hc["available"] is True
    assert hc["domain"] == "hiring"
    assert 2 <= len(hc["patterns"]) <= 4
    assert hc["defaultAttributes"] and hc["checkPriorities"]
    assert any("Bertrand and Mullainathan" in c for c in hc["citations"])
    assert any("Amazon" in c for c in hc["citations"])


def test_historical_context_unknown_domain(unknown_domain):
    hc = unknown_domain["historicalContext"]
    assert hc["available"] is False
    assert hc["reason"]


def test_historical_context_lending(credit_eu):
    hc = credit_eu["historicalContext"]
    assert hc["available"] is True
    assert hc["domain"] == "lending"
    assert any("Bartlett" in c for c in hc["citations"])


# ── G-19: harm-direction routing ────────────────────────────────────────────


def test_punitive_routes_headline_to_predictive_parity(hiring_us):
    rec = hiring_us["recommendedDefinition"]
    assert rec["headlineMetric"] == "predictive_parity"
    defs = [d["definition"] for d in rec["definitionsApplied"]]
    assert any("predictive parity" in d.lower() for d in defs)
    assert rec["rejected"], "the de-emphasized family must be disclosed"
    assert all(d.get("why") for d in rec["definitionsApplied"])


def test_assistive_routes_headline_to_equal_opportunity(credit_eu):
    rec = credit_eu["recommendedDefinition"]
    assert rec["headlineMetric"] == "equal_opportunity"


def test_scores_exposed_adds_calibration(hiring_us, credit_eu):
    for data in (hiring_us, credit_eu):
        defs = [
            d["definition"].lower() for d in data["recommendedDefinition"]["definitionsApplied"]
        ]
        assert any("calibration" in d for d in defs)


def test_impossibility_acknowledged_when_base_rates_differ(credit_eu):
    # calibration (scores_exposed) + error-rate parity (assistive) are
    # both applied and the fixture's base rates differ (~0.7 vs ~0.4):
    # the trade-off must state the impossibility and the chosen emphasis.
    trade = credit_eu["recommendedDefinition"]["tradeoff"]
    assert "base rates" in trade
    assert "Chouldechova" in trade
    assert "equal opportunity" in trade.lower()


def test_legacy_recommendation_keys_still_present(hiring_us):
    rec = hiring_us["recommendedDefinition"]
    for key in ("primary", "secondaries", "rationale", "howToApply", "labelFree"):
        assert key in rec, f"legacy key {key} must survive (additive only)"


# ── G-32: reference-group override + targets ────────────────────────────────


def test_reference_default_is_majority_group(unknown_domain):
    row = _pv_row(unknown_domain, "group")
    assert row["referenceGroup"] == "A"
    assert row["referenceOverrideHonored"] is False
    rg = unknown_domain["referenceGroups"]["group"]
    assert rg["requested"] is None
    assert rg["used"] == "A"
    assert "No reference group was requested" in rg["note"]


def test_reference_override_honored_and_disclosed(ref_override_b):
    row = _pv_row(ref_override_b, "group")
    assert row["referenceGroup"] == "B"
    assert row["referenceOverrideHonored"] is True
    assert row["worstGroup"] == "C"
    rg = ref_override_b["referenceGroups"]["group"]
    assert rg["requested"] == "B"
    assert rg["used"] == "B"
    assert "was used as the comparison baseline" in rg["note"]


def test_reference_override_unavailable_disclosed(ref_override_missing):
    row = _pv_row(ref_override_missing, "group")
    assert row["referenceOverrideHonored"] is False
    assert row["referenceGroup"] == "A"  # fell back to the default
    rg = ref_override_missing["referenceGroups"]["group"]
    assert rg["requested"] == "Z"
    assert rg["used"] == "A"
    assert "not found" in rg["note"]


def test_targets_pass_flags_against_custom_ratio(hiring_us):
    targets = hiring_us["targets"]
    assert targets["targetRatio"] == pytest.approx(0.9)
    rows = {r["attribute"]: r for r in targets["perAttribute"]}
    assert "gender" in rows
    # female/male selection ratio ~0.29 < 0.9 -> must fail the target.
    assert rows["gender"]["passes"] is False
    assert rows["gender"]["fourFifthsRatio"] < 0.9
    assert targets["note"]


def test_targets_ratio_clamped_to_band():
    assert build_targets([], 0.3)["targetRatio"] == pytest.approx(0.5)
    assert build_targets([], 1.7)["targetRatio"] == pytest.approx(1.0)
    assert build_targets([], None)["targetRatio"] == pytest.approx(0.8)
    assert "clamped" in build_targets([], 0.3)["note"]


def test_targets_pass_flag_true_when_ratio_clears():
    fake_rows = [{"attribute": "x", "assessable": True, "fourFifthsRatio": 0.92}]
    t = build_targets(fake_rows, 0.9)
    assert t["perAttribute"][0]["passes"] is True


# ── G-34: recheck window + secret-free suite ────────────────────────────────


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_recheck_plus_12_months_when_ll144_applies(hiring_us):
    rc = hiring_us["recheck"]
    assessed = _parse_iso(rc["assessedAt"])
    valid = _parse_iso(rc["validUntil"])
    assert valid == _add_months(assessed, 12)
    assert rc["basis"]


def test_recheck_plus_6_months_otherwise(credit_eu, unknown_domain):
    for data in (credit_eu, unknown_domain):
        rc = data["recheck"]
        assessed = _parse_iso(rc["assessedAt"])
        valid = _parse_iso(rc["validUntil"])
        assert valid == _add_months(assessed, 6)


def test_recheck_suite_strips_planted_secrets(hiring_us):
    suite = hiring_us["recheck"]["suite"]
    blob = json.dumps(suite)
    for secret_key in ("auth_token", "model_base64", "model_envelope", "org_api_key"):
        assert secret_key not in suite
        assert secret_key not in blob
    for secret_value in ("sekrit-token-123", "QUJDREVG", "org-sekrit-456"):
        assert secret_value not in blob
    # The re-runnable parts survive.
    assert suite["domain"] == "hiring"
    assert suite["protected_attributes"] == ["gender"]
    assert suite["target_ratio"] == pytest.approx(0.9)


def test_recheck_suite_strips_nested_llm_config_token():
    rc = build_recheck(
        {
            "domain": "d",
            "llm_config": {
                "endpoint_url": "http://example.invalid",
                "auth_token": "tok-999",
                "model_name": "m",
            },
        },
        ll144_applicable=False,
    )
    cfg = rc["suite"]["llm_config"]
    assert "auth_token" not in cfg
    assert cfg["endpoint_url"] == "http://example.invalid"
    assert "tok-999" not in json.dumps(rc["suite"])


# ── envelope: scope line + JSON safety ──────────────────────────────────────


def test_scope_names_regulatory_packaging(hiring_us):
    assert any("Regulatory packaging" in c for c in hiring_us["scope"]["covered"])


def test_result_remains_json_serialisable(hiring_us, credit_eu):
    for data in (hiring_us, credit_eu):
        json.dumps(data)  # must not raise
