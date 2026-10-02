"""Pins for three Pulse defects measured on the published test objects, 2026-10-01.

1. A CLEAN control read "Not fit to deploy". On data with no planted bias
   (docs/site/test-objects/data/synthetic/clean.csv), the family-wise correction
   re-confirmed national origin after the omnibus gate had rejected it (omnibus
   p 0.31), age entered the correction at a selected-pair p of 0.006 although its
   test across all age bands gave 0.025, and unconfirmed gaps were toned critical
   by the gap size or the four-fifths ratio alone. 29 CFR 1607.4(D), the rule the
   US-employment screen cites: "Greater differences in selection rate may not
   constitute adverse impact where the differences are based on small numbers and
   are not statistically significant."
2. A measured agent NULL read "Insufficient assessable data": the agent route gave
   build_assurance_verdict an empty per_variable, which its BGL-3 guard reads as
   "nothing was assessed".
3. On pandas 3 the agent probe failed on any export where an episode calls no
   tool: astype(str) leaves NaN a float there, and sorting tool names raised.

Each pin was checked against the unfixed code and failed there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import orchestrator as o
from vfairness.operations.pulse.agent_probe import NO_ACTION_LABEL, agent_probe_pulse


# ---------------------------------------------------------------- 1. false alarms
def test_an_unconfirmed_disparity_is_never_critical():
    assert o._cap_unconfirmed("critical", False) == "warn"
    assert o._cap_unconfirmed("critical", True) == "critical"
    assert o._cap_unconfirmed("warn", False) == "warn"
    assert o._cap_unconfirmed("pass", False) == "pass"


def _three_group_frame(rates, n_per_group, seed=0):
    rng = np.random.default_rng(seed)
    groups, preds = [], []
    for label, (rate, n) in zip("ABC", zip(rates, n_per_group)):
        k = int(round(rate * n))
        p = np.array([1] * k + [0] * (n - k))
        rng.shuffle(p)
        groups += [label] * n
        preds += p.tolist()
    return pd.DataFrame({"g": groups}), np.array(preds)


def test_the_per_variable_test_records_the_omnibus_p_for_more_than_two_groups():
    df, y = _three_group_frame([0.30, 0.30, 0.10], [200, 200, 20])
    row = o._per_variable(df, "g", y, None)
    assert isinstance(row.get("pValueOmnibus"), float), row.get("pValueOmnibus")
    assert row["pValueOmnibus"] >= 0.0


def test_a_small_extreme_group_is_not_critical_without_confirmation():
    # 20 people at 10 percent against 30 percent: ratio 0.33, a gap of 20 points,
    # US employment. Not significant across the three groups at this size, so it
    # may be a watch item, never critical.
    df, y = _three_group_frame([0.30, 0.30, 0.10], [200, 200, 20])
    fw = o._legal_screen_framework("recruitment / employment screening", "US")
    row = o._per_variable(df, "g", y, None, framework=fw)
    if not row["significant"]:
        assert row["tone"] != "critical", row


@pytest.mark.slow
def test_the_clean_synthetic_control_is_not_reported_as_unfit():
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs/site/test-objects/data/synthetic/clean.csv"
    )
    df = pd.read_csv(path).drop(columns=["true_qualification_score", "_bias_flags"])
    attrs = [
        "gender",
        "race_ethnicity",
        "age",
        "disability_status",
        "national_origin",
        "religion",
        "marital_status",
        "sexual_orientation",
        "veteran_status",
        "primary_language",
    ]
    res = o.run_pulse(
        df,
        {
            "source_kind": "predictive",
            "domain": "recruitment / employment screening",
            "jurisdiction": "US",
            "protected_attributes": attrs,
        },
    )["data"]
    confirmed = [p["attribute"] for p in res["perVariable"] if p.get("significant")]
    critical = [p["attribute"] for p in res["perVariable"] if p.get("tone") == "critical"]
    assert confirmed == [], f"nothing was planted, yet confirmed: {confirmed}"
    assert critical == [], f"nothing was planted, yet critical: {critical}"
    # The correction may only withdraw a confirmation, never grant one.
    for p in res["perVariable"]:
        if p.get("significant"):
            assert p.get("significantRaw"), p


# ------------------------------------------------------------ 2. agent nulls
def _episodes(per_group: int, biased: bool, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in ["a", "b", "c"]:
        p_approve = 0.2 if (biased and g == "c") else 0.8
        for i in range(per_group):
            tool = "approve" if rng.random() < p_approve else "refer"
            rows.append({"trace_id": f"{g}-{i}", "group": g, "tool": tool})
    return pd.DataFrame(rows)


def test_a_measured_agent_null_is_an_opinion_not_a_disclaimer():
    out = agent_probe_pulse(_episodes(120, biased=False), {}, "tool", "group", "lending", "")
    data = out["data"]
    assert data["agent"]["sampleAdequacy"]["adequate"] is True
    assert data["assurance"]["overall"] != "Disclaimer", data["verdict"]["headline"]
    assert "Insufficient assessable data" not in data["verdict"]["headline"]


def test_an_agent_below_the_sample_floor_is_still_a_disclaimer():
    out = agent_probe_pulse(_episodes(5, biased=False), {}, "tool", "group", "lending", "")
    assert out["data"]["assurance"]["overall"] == "Disclaimer"


def test_a_biased_agent_is_still_found():
    out = agent_probe_pulse(_episodes(120, biased=True), {}, "tool", "group", "lending", "")
    assert out["data"]["assurance"]["overall"] in ("Adverse", "Qualified"), out["data"]["verdict"]


# ------------------------------------------------------ 3. missing tool, any pandas
def test_an_episode_without_a_tool_is_counted_under_an_explicit_label():
    df = _episodes(60, biased=False)
    df.loc[df.index % 7 == 0, "tool"] = np.nan
    # pandas >= 2.3 has the NaN-valued string dtype that pandas 3 uses by default;
    # older pandas (down to the declared floor) keeps the same data as an object
    # column holding NaN. Both are the "missing tool" case and both must work.
    try:
        string_with_nan = pd.StringDtype("python", na_value=np.nan)
    except TypeError:
        string_with_nan = None
    if string_with_nan is not None:
        df["tool"] = df["tool"].astype(string_with_nan)
    out = agent_probe_pulse(df, {}, "tool", "group", "lending", "")
    omnibus = out["data"]["agent"]["omnibus"]
    assert omnibus.get("available") is True, omnibus
    assert NO_ACTION_LABEL in omnibus["tools"], omnibus["tools"]
    assert "nan" not in omnibus["tools"], omnibus["tools"]
