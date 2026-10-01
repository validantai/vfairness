"""Audit-3 pin: the Pulse deploy verdict must be driven by the worst
per-attribute SEVERITY (tone), not the largest absolute gap.

Regression finding (third-iteration audit, orchestrator.py:5820): the
headline/deploy verdict took ranked[0] after sorting assessable attributes by
absolute gap descending. But a row's tone is NOT monotonic in the gap:
_disparity_tone escalates a reliable group selected 0% of the time (total
exclusion, four-fifths 0) to 'critical' at a SMALL absolute gap, while a larger
but benign gap stays 'warn'. So a critical total-exclusion on one attribute was
hidden behind a larger but benign gap on another, yielding the reassuring
"Deployable only with monitoring" headline for a system that totally excludes a
group. The fix drives the overall tone from the MAX tone across all assessable
attributes (Kearns 2018 worst-case subgroup fairness) and picks the narrative
row by (tone-rank, gap), gap only as a within-tone tiebreak. `ranked` stays
gap-sorted for the display list.

Literature: Kearns, Neel, Roth & Wu 2018, "Preventing Fairness Gerrymandering"
(ICML): subgroup fairness is the worst-CASE violation over subgroups; EEOC
four-fifths (total exclusion = ratio 0 = the single most severe finding).
"""

import numpy as np
import pandas as pd

from vfairness.operations.pulse import run_pulse


def _feature(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _cells_df(attr_x: str, attr_y: str, cells, feature=None) -> pd.DataFrame:
    """Build a 2-attribute frame from a 2x2 (or larger) contingency with EXACT
    positive counts per cell, so BOTH marginal selection rates are exact and
    fully deterministic (no RNG).

    ``cells`` is a list of ``(x_label, y_label, n, n_pos)`` tuples.
    ``feature`` is an optional callable ``n -> array`` for the numeric feature
    column; the default monotone ramp aligns with the cell block order, which
    is uncorrelated enough here that it stays below the proxy floor.
    """
    xs: list = []
    ys: list = []
    preds: list = []
    for xl, yl, n, npos in cells:
        assert 0 <= npos <= n
        xs += [xl] * n
        ys += [yl] * n
        preds += [1] * npos + [0] * (n - npos)
    total = len(xs)
    feat = (feature or _feature)(total)
    return pd.DataFrame(
        {attr_x: xs, attr_y: ys, "prediction": np.asarray(preds, dtype=int), "feature": feat}
    )


def _defect_df() -> pd.DataFrame:
    """gender: A 0.30 / B 0.15 (gap 0.15, four-fifths 0.50 -> WARN).
    region: X 0.09 (majority ref) / Z 0.5175 / Y 0.00 (total exclusion,
    four-fifths 0 -> CRITICAL) with gap 0.09 < gender's 0.15.

    A gap-sorted headline picks gender (warn) and hides the region exclusion.
    """
    n = 1200
    gender = np.array(["A"] * 600 + ["B"] * 600)
    pred = np.zeros(n, dtype=int)
    pred[0:180] = 1  # A: 180/600 = 0.30
    pred[600:690] = 1  # B: 90/600 = 0.15  (270 positives total)

    zero_idx = list(np.where(pred == 0)[0])
    one_idx = list(np.where(pred == 1)[0])
    region = np.array(["?"] * n, dtype=object)
    for i in zero_idx[:100]:  # Y: 100 zeros -> rate 0.0 (total exclusion)
        region[i] = "Y"
    rem_zeros = zero_idx[100:]
    for i in one_idx[:63] + rem_zeros[:637]:  # X: 63/700 = 0.09 (majority ref)
        region[i] = "X"
    for i in one_idx[63:] + rem_zeros[637:]:  # Z: 207/400 = 0.5175 (best rate)
        region[i] = "Z"
    assert (region == "?").sum() == 0
    return pd.DataFrame(
        {"gender": gender, "region": region, "prediction": pred, "feature": _feature(n)}
    )


def _tone_of(data: dict, attribute: str) -> str:
    for r in data["perVariable"]:
        if r.get("assessable") and r.get("attribute") == attribute:
            return r["tone"]
    raise AssertionError(f"{attribute} not assessable")


# ── the defect: a small-gap total-exclusion CRITICAL must beat a larger
#    benign WARN gap (the exact scenario that used to give the wrong answer) ──


def test_critical_total_exclusion_beats_larger_benign_gap():
    d = run_pulse(_defect_df(), {"protected_attributes": ["gender", "region"]})["data"]

    # Setup sanity: the more-severe attribute (region, total exclusion) has the
    # SMALLER absolute gap; the benign attribute (gender) has the larger gap.
    assert _tone_of(d, "region") == "critical"
    assert _tone_of(d, "gender") == "warn"
    by_attr = {r["attribute"]: r for r in d["perVariable"] if r.get("assessable")}
    assert by_attr["region"]["gap"] < by_attr["gender"]["gap"]

    v = d["verdict"]
    # The headline/deploy verdict must reflect the worst SEVERITY, not the
    # largest gap. Before the fix: tone 'warn' / "Deployable only with
    # monitoring: watch-level gap for B".
    assert v["tone"] == "critical", v
    assert v["headline"].startswith("Not fit to deploy as-is"), v["headline"]
    assert 'for "Y"' in v["headline"], v["headline"]  # names the excluded group
    assert 'in "region"' in v["summary"], v["summary"]

    # `ranked` stays gap-sorted for the DISPLAY list (gender's 0.15 first),
    # even though it no longer drives the headline.
    ranked = v["ranked"]
    assert [r["attribute"] for r in ranked] == ["gender", "region"], ranked
    assert ranked[0]["tone"] == "warn" and ranked[1]["tone"] == "critical"


# ── does-not-overcorrect cases ───────────────────────────────────────────────


def test_largest_gap_that_is_also_most_severe_still_wins():
    """When the biggest-gap attribute IS the most severe (a large, confirmed
    gap that is critical on its own), it must still drive the headline. Guards
    against the fix inverting the selection toward the smallest gap.

    big: X 0.80 / Y 0.20 -> gap 0.60 (CRITICAL). small: A 0.56 / B 0.44 ->
    gap 0.12 (WARN). Both averages 0.50 so the single prediction column is
    consistent. The critical attribute is also the largest-gap one.
    """
    df = _cells_df(
        "big",
        "small",
        [
            ("X", "A", 500, 460),  # XA rate 0.92
            ("X", "B", 500, 340),  # XB rate 0.68
            ("Y", "A", 500, 100),  # YA rate 0.20
            ("Y", "B", 500, 100),  # YB rate 0.20
        ],
    )
    d = run_pulse(df, {"protected_attributes": ["big", "small"]})["data"]
    assert _tone_of(d, "big") == "critical"
    assert _tone_of(d, "small") == "warn"
    v = d["verdict"]
    assert v["tone"] == "critical", v
    assert v["headline"].startswith("Not fit to deploy as-is"), v["headline"]
    assert 'for "Y"' in v["headline"], v["headline"]
    assert 'in "big"' in v["summary"], v["summary"]


def test_within_tone_tiebreak_prefers_the_larger_gap():
    """When two attributes share the top tone (both WARN), the larger absolute
    gap breaks the tie and names the headline attribute. Guards the gap
    tiebreak inside a tone.

    a1: A 0.575 / B 0.425 -> gap 0.15 (WARN). a2: P 0.55 / Q 0.45 -> gap 0.10
    (WARN). Neither is critical; a1 has the larger gap.
    """
    df = _cells_df(
        "a1",
        "a2",
        [
            ("A", "P", 500, 300),  # 0.60
            ("A", "Q", 500, 275),  # 0.55
            ("B", "P", 500, 250),  # 0.50
            ("B", "Q", 500, 175),  # 0.35
        ],
    )
    d = run_pulse(df, {"protected_attributes": ["a1", "a2"]})["data"]
    assert _tone_of(d, "a1") == "warn"
    assert _tone_of(d, "a2") == "warn"
    v = d["verdict"]
    assert v["tone"] == "warn", v
    assert v["headline"].startswith("Deployable only with monitoring"), v["headline"]
    assert 'for "B"' in v["headline"], v["headline"]  # a1's worst group
    assert 'in "a1"' in v["summary"], v["summary"]


def test_all_pass_stays_within_budget_no_spurious_escalation():
    """When no attribute is severe, the max-tone roll-up must NOT invent
    severity: the verdict stays 'pass' / within budget.

    a1: A 0.52 / B 0.48 (gap 0.04). a2: P 0.51 / Q 0.49 (gap 0.02). Both far
    inside the four-fifths band; both PASS.
    """
    # Uncorrelated (seeded-noise) feature so the run is genuinely clean: no
    # stage degrades, so any escalation would have to come from the verdict
    # roll-up itself.
    df = _cells_df(
        "a1",
        "a2",
        [
            ("A", "P", 500, 260),  # 0.52
            ("A", "Q", 500, 260),
            ("B", "P", 500, 250),  # 0.50 / 0.46
            ("B", "Q", 500, 230),
        ],
        feature=lambda n: np.random.default_rng(3).normal(size=n),
    )
    d = run_pulse(df, {"protected_attributes": ["a1", "a2"]})["data"]
    assert _tone_of(d, "a1") == "pass"
    assert _tone_of(d, "a2") == "pass"
    v = d["verdict"]
    # No degradations expected on this clean two-group frame; if a stage had
    # failed, the run-level guard would bump the tone to >= warn.
    assert not d.get("degradations"), d.get("degradations")
    assert v["tone"] == "pass", v
    assert v["headline"] == "Within fairness budget on the assessed attributes.", v["headline"]


# ── ADDITIONAL in-file fix (not the assigned finding): the proxy-chain
#    synthesis referenced an unbound `strength` (typo for `p_strength`) and
#    crashed the whole proxies stage, which the run-level guard then turned
#    into a spurious verdict downgrade. See orchestrator.py:1853. ────────────


def test_proxy_chain_synthesis_does_not_crash_and_downgrade_verdict():
    """A feature strongly correlated with a protected attribute produces a
    proxy with strength >= 0.15, which drives the chain-synthesis loop that
    formatted an UNBOUND local (`strength`). Before the fix that raised
    UnboundLocalError, _safe recorded a 'proxies' degradation, and the
    run-level guard bumped an otherwise-clean 'pass' verdict to 'warn'.
    """
    # a1 blocks perfectly track the feature ramp -> eta^2 ~ 1.0 (proxy).
    df = _cells_df(
        "a1",
        "a2",
        [
            ("A", "P", 400, 208),  # a1: A ~0.52, a2 balanced -> all PASS
            ("A", "Q", 400, 208),
            ("B", "P", 400, 200),
            ("B", "Q", 400, 184),
        ],
        feature=lambda m: np.linspace(0.0, 1.0, m),  # monotone -> tracks a1 block order
    )
    d = run_pulse(df, {"protected_attributes": ["a1", "a2"]})["data"]

    # The proxies stage must NOT be in degradations (it crashed before).
    assert "proxies" not in [g.get("stage") for g in (d.get("degradations") or [])], d.get(
        "degradations"
    )
    # And a strong feature->a1 proxy was actually detected (the path that
    # reaches the previously-crashing chain-synthesis code).
    prox = d.get("proxies") or {}
    assert prox.get("available") is not False
    feats = {p.get("feature") for p in (prox.get("proxies") or [])}
    assert "feature" in feats, prox.get("proxies")
    # With the crash gone, a clean all-pass frame stays 'pass' (no spurious
    # downgrade) even though the feature is a strong proxy.
    assert _tone_of(d, "a1") == "pass"
    assert _tone_of(d, "a2") == "pass"
    assert d["verdict"]["tone"] == "pass", d["verdict"]
