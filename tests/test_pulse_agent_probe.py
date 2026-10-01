"""Contract tests for the Pulse agent-trace probe (agent_probe_pulse).

Pulse close plan Phase 3: G-09 (all-groups comparison, omnibus gate,
family-wise Benjamini-Hochberg correction, effect-size severity from
Cramér's V, bootstrap CIs, sample-adequacy contract field, trajectory
shape) and G-33 (delegation routing via DelegationRoutingAuditor).

Every call routes through run_pulse with source_kind='agent' so the
router integration is covered. Deterministic: fixtures are either exact
counts or seeded (seed 7), and the probe's own resampling is seeded.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse

SEED = 7


def _pulse(df):
    return run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})["data"]


def _comparison(agent, ga, gb):
    for c in agent["perComparison"]:
        if {c["groupA"], c["groupB"]} == {ga, gb}:
            return c
    raise AssertionError(f"missing comparison {ga} vs {gb}")


# ── fixtures ────────────────────────────────────────────────────────────────


def _three_group_df():
    """Exact-count fixture, n=120 per group. Groups A and B share the
    same tool distribution; group C is planted with a strong skew toward
    'escalate'. For the A vs C table the uncorrected chi-square is 43.2
    on n=240, so Cramér's V = sqrt(43.2/240) = 0.424 -> severity 'warn'.
    """
    base = {"search": 84, "escalate": 24, "respond": 12}
    skew = {"search": 36, "escalate": 72, "respond": 12}
    rows = []
    for g, dist in (("A", base), ("B", base), ("C", skew)):
        for tool, k in dist.items():
            rows.extend({"group": g, "tool": tool} for _ in range(k))
    return pd.DataFrame(rows)


def _five_null_groups_df():
    """Five groups drawn (seeded) from the SAME tool distribution: the
    max pairwise gap grows with the number of groups even under the
    null, which is exactly what the omnibus gate + BH must absorb."""
    rng = np.random.default_rng(SEED)
    tools = np.array(["search", "escalate", "respond"])
    frames = []
    for g in ("A", "B", "C", "D", "E"):
        picks = rng.choice(tools, size=120, p=[0.6, 0.3, 0.1])
        frames.append(pd.DataFrame({"group": g, "tool": picks}))
    return pd.concat(frames, ignore_index=True)


def _no_extra_columns_df():
    """Two balanced groups, identical tool use, no trajectory or
    delegation columns."""
    return pd.DataFrame({"group": ["A"] * 30 + ["B"] * 30, "tool": ["search", "escalate"] * 30})


# ── G-09: all-groups statistics ─────────────────────────────────────────────


def test_three_groups_planted_skew_fires_family_wise():
    d = _pulse(_three_group_df())
    agent = d["agent"]
    assert agent["available"] is True
    assert agent["sampleAdequacy"]["adequate"] is True

    # Omnibus chi-square on the full group x tool table is significant.
    assert agent["omnibus"]["significant"] is True
    assert agent["omnibus"]["pValue"] < 0.05

    # All three pairs were compared, not just the two most-populous.
    assert len(agent["perComparison"]) == 3

    # The planted group's comparisons carry family-wise significant
    # per-tool results; the null pair (A vs B) carries none.
    ac = _comparison(agent, "A", "C")
    assert any(t["family_significant"] for t in ac["tools"])
    ab = _comparison(agent, "A", "B")
    assert not any(t["family_significant"] for t in ab["tools"])

    tool_findings = [f for f in d["bias"] if f["type"] == "agent_tool_bias"]
    assert tool_findings
    # Every finding names the planted group C, never the null pair.
    assert all("C" in f["plain"] for f in tool_findings)

    # Severity comes from Cramér's V, not a hardcoded "high":
    # V = 0.424 for the planted pair, so severity is exactly "warn".
    assert ac["cramersV"] == pytest.approx(0.4243, abs=0.01)
    assert ac["severity"] == "warn"
    assert all(f["severity"] == "warn" for f in tool_findings)
    assert all(f["severity"] != "high" for f in d["bias"])

    # Bootstrap CI on the largest per-tool rate difference is reported
    # and excludes zero for the planted pair.
    largest = ac["largestRateDiff"]
    assert largest["resamples"] == 1000 and largest["seed"] == 7
    assert largest["ciLow"] * largest["ciHigh"] > 0


def test_five_null_groups_no_finding_fires():
    d = _pulse(_five_null_groups_df())
    agent = d["agent"]
    # 5 groups -> all 10 pairs compared.
    assert len(agent["perComparison"]) == 10
    # The omnibus gate + family-wise BH hold: no bias finding fires on
    # same-distribution groups.
    assert [f for f in d["bias"] if f["type"] in ("agent_tool_bias", "agent_action_bias")] == []
    assert d["assurance"]["overall"] in ("Unqualified", "Disclaimer")


def test_low_n_honest_disclaimer():
    rows = []
    for g in ("A", "B", "C"):
        for i in range(8):
            rows.append({"group": g, "tool": "search" if i % 2 else "escalate"})
    d = _pulse(pd.DataFrame(rows))
    sa = d["agent"]["sampleAdequacy"]
    assert sa["adequate"] is False
    assert sa["floor"] == 20
    assert sa["perGroup"] == {"A": 8, "B": 8, "C": 8}
    assert d["agent"]["available"] is False
    assert d["agent"]["perComparison"] == []
    assert d["assurance"]["overall"] == "Disclaimer"
    assert d["verdict"]["tone"] == "disclaimer"
    assert d["bias"] == []


def test_partial_low_n_excluded_and_named():
    df = pd.concat(
        [_three_group_df(), pd.DataFrame({"group": ["D"] * 8, "tool": ["search"] * 8})],
        ignore_index=True,
    )
    d = _pulse(df)
    sa = d["agent"]["sampleAdequacy"]
    assert sa["adequate"] is False
    assert sa["perGroup"]["D"] == 8
    assert "D" in sa["note"]
    # D is excluded from the comparisons; the three adequate groups
    # still yield their three pairs.
    assert len(d["agent"]["perComparison"]) == 3
    for c in d["agent"]["perComparison"]:
        assert "D" not in (c["groupA"], c["groupB"])


# ── G-09.5: trajectory shape ────────────────────────────────────────────────


def test_trajectory_steps_gap_fires():
    rng = np.random.default_rng(SEED)
    rows = []
    for g, mean in (("A", 5.0), ("B", 10.0)):
        for st in rng.normal(mean, 1.0, 60):
            rows.append({"group": g, "tool": "search", "steps": float(max(st, 1.0))})
    d = _pulse(pd.DataFrame(rows))
    tr = d["agent"]["trajectory"]
    assert tr["available"] is True
    steps = [m for m in tr["perMetric"] if m["metric"] == "steps"]
    assert steps and steps[0]["kind"] == "mean"
    comp = steps[0]["comparisons"][0]
    assert comp["significant"] is True
    assert comp["pAdjusted"] < 0.05
    findings = [f for f in d["bias"] if f["type"] == "agent_trajectory_bias"]
    assert findings
    # A 5-vs-10-step gap at sd 1 is a huge standardized effect.
    assert findings[0]["severity"] == "critical"


def test_trajectory_absent_not_applicable():
    d = _pulse(_no_extra_columns_df())
    tr = d["agent"]["trajectory"]
    assert tr["available"] is False
    assert tr["notApplicable"] is True
    assert [f for f in d["bias"] if f["type"] == "agent_trajectory_bias"] == []


# ── G-33: delegation routing ────────────────────────────────────────────────


def test_delegation_skew_fires():
    rows = []
    # Identical tool use; delegation strongly depends on the group
    # (senior 48/60 for A vs 12/60 for B): uncorrected chi-square 43.2
    # on n=120, so Cramér's V = 0.6 -> severity 'critical'.
    for g, n_senior in (("A", 48), ("B", 12)):
        for i in range(60):
            rows.append(
                {
                    "group": g,
                    "tool": "search",
                    "delegate": ("senior_agent" if i < n_senior else "junior_agent"),
                }
            )
    d = _pulse(pd.DataFrame(rows))
    dg = d["agent"]["delegation"]
    assert dg["available"] is True
    assert dg["column"] == "delegate"
    assert dg["significant"] is True
    assert dg["cramersV"] == pytest.approx(0.6, abs=0.01)
    findings = [f for f in d["bias"] if f["type"] == "delegation_routing"]
    assert findings
    assert findings[0]["severity"] == "critical"


def test_delegation_absent_not_applicable():
    d = _pulse(_no_extra_columns_df())
    dg = d["agent"]["delegation"]
    assert dg["available"] is False
    assert dg["notApplicable"] is True
    assert "no route/delegation column" in dg["reason"]
    assert [f for f in d["bias"] if f["type"] == "delegation_routing"] == []
