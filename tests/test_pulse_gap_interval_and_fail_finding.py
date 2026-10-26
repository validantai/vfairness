"""Pins for two Pulse statistics defects measured on the platform's recruitment
sample (public/sample-data/recruitment_fairness_dataset.csv, 2,700 rows, hiring,
EU), 2026-10-09.

1. A FALSELY NARROW GAP INTERVAL. 0 of 22 Native American applicants invited
   against 114 of 1,601 White read "95% CI 5.8 to 8.4 points", statistically
   confirmed after FDR correction (adjusted p 0.009), while the permutation test
   across race_ethnicity read p 0.70. The interval was a percentile bootstrap: a
   group with no positives resamples only zeros, so it added no variance and the
   interval ignored the uncertainty of the small group's rate. The gap interval
   is now the Newcombe hybrid score interval (Newcombe 1998, method 10), and the
   p-value that enters the Benjamini-Hochberg family is its dual, so the
   confirmation can never contradict the printed interval.

2. A FAIL WITH NO FINDING. Age was graded critical ("Fail" on the report's
   four-fifths row: 0 of 318 applicants aged 18 to 24 invited, ratio 0.00,
   statistically confirmed) but build_assurance_verdict raised its
   disparate-impact finding only for a gap of 5 points or more outside US
   employment, and age's gap to its reference was 4.6 points. The report then
   graded a metric Fail that no finding and no recommendation explained.

3. THE PAIR IS CHOSEN, SO IT IS CORRECTED FOR THE CHOICE. Found while proving 1:
   with an honest interval, the published clean control (nothing planted) read
   age as confirmed (9 of 56 against 45 of 118, pair p 0.0034, BH 0.034), which
   the bootstrap had hidden only through its 1/1000 resolution. The worst group
   is the most extreme of several comparisons with the reference, so its p is
   Bonferroni-corrected for that choice before it can confirm or enter the BH
   family. The clean control is quiet again, the planted mechanisms main
   confirmed are still confirmed, and main's three spurious confirmations on the
   synthetic pack are gone.

Each pin below was run against the unfixed code (origin/main 55073b0) and failed
there, except the controls, which are marked as such and pass on both.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    _newcombe_hybrid_score_difference,
)
from vfairness.exceptions import ConfigurationError
from vfairness.operations.pulse import orchestrator as o
from vfairness.operations.reporting import build_assurance_verdict

# ------------------------------------------------------------------ helpers


def _two_groups(ref: tuple, worst: tuple, seed: int = 0):
    """A frame with exactly two groups and exact selection counts.

    ``ref`` and ``worst`` are (label, n, selected)."""
    rng = np.random.default_rng(seed)
    labels, decisions = [], []
    for label, n, k in (ref, worst):
        d = np.array([1] * k + [0] * (n - k))
        rng.shuffle(d)
        labels += [label] * n
        decisions += d.tolist()
    return pd.DataFrame({"g": labels}), np.asarray(decisions)


def _groups(*specs, seed: int = 0):
    """A frame with exact selection counts per group; specs are (label, n, selected)."""
    rng = np.random.default_rng(seed)
    labels, decisions = [], []
    for label, n, k in specs:
        d = np.array([1] * k + [0] * (n - k))
        rng.shuffle(d)
        labels += [label] * n
        decisions += d.tolist()
    return pd.DataFrame({"g": labels}), np.asarray(decisions)


def _wald(x1, n1, x2, n2, z=1.959963984540054):
    p1, p2 = x1 / n1, x2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) - z * se, (p1 - p2) + z * se


# ------------------------------------------------- 1a. the interval method itself

# Newcombe (1998) Table II, method 10 (hybrid score), to the four decimals the
# paper prints. statsmodels' confint_proportions_2indep(method="newcomb")
# returns the same values; these are the published ones.
_NEWCOMBE_TABLE_II = [
    ((56, 70, 48, 80), (0.0524, 0.3339)),
    ((9, 10, 3, 10), (0.1705, 0.8090)),
    ((5, 56, 0, 29), (-0.0381, 0.1926)),
    ((0, 10, 0, 20), (-0.1611, 0.2775)),
    ((0, 10, 0, 10), (-0.2775, 0.2775)),
    ((10, 10, 0, 20), (0.6791, 1.0000)),
    ((10, 10, 0, 10), (0.6075, 1.0000)),
]


@pytest.mark.parametrize("counts,expected", _NEWCOMBE_TABLE_II)
def test_the_gap_interval_reproduces_newcombe_1998_table_ii(counts, expected):
    d, lo, hi, _p = _newcombe_hybrid_score_difference(*counts)
    assert d == pytest.approx(counts[0] / counts[1] - counts[2] / counts[3])
    assert round(lo, 4) == pytest.approx(expected[0], abs=1e-12)
    assert round(hi, 4) == pytest.approx(expected[1], abs=1e-12)


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_the_p_value_is_the_dual_of_the_interval(level):
    """Confirmed by the p-value exactly when the interval excludes zero, on every
    count pair of a grid that includes zero and full counts and groups of 5."""
    alpha = 1.0 - level
    checked = 0
    for n1 in (5, 9, 22, 60, 300):
        for n2 in (5, 12, 22, 120):
            for x1 in sorted({0, 1, n1 // 3, n1 // 2, n1 - 1, n1}):
                for x2 in sorted({0, 1, n2 // 4, n2 // 2, n2 - 1, n2}):
                    _d, lo, hi, p = _newcombe_hybrid_score_difference(x1, n1, x2, n2, level)
                    assert 0.0 <= p <= 1.0
                    assert (p < alpha) == (lo > 0 or hi < 0), (x1, n1, x2, n2, lo, hi, p)
                    checked += 1
    assert checked > 300


def test_equal_rates_read_p_one_and_an_impossible_table_is_refused():
    assert _newcombe_hybrid_score_difference(3, 10, 6, 20)[3] == 1.0
    for bad in [(5, 4, 1, 10), (-1, 10, 1, 10), (1, 0, 1, 10), (1, 10, 11, 10)]:
        with pytest.raises(ConfigurationError):
            _newcombe_hybrid_score_difference(*bad)


def test_a_p_value_below_the_smallest_double_is_reported_as_zero():
    # 1000 of 1000 against 0 of 1000: the interval reaches zero only near
    # z = 49, where the two-sided normal tail underflows a double, so the dual
    # p is reported as 0.0, not as a made-up floor.
    d, lo, hi, p = _newcombe_hybrid_score_difference(1000, 1000, 0, 1000)
    assert d == 1.0 and lo > 0.99 and hi == 1.0
    assert p == 0.0


# ----------------------------------------- 1b. the interval Pulse puts on the gap


def test_a_zero_count_group_no_longer_gets_a_falsely_narrow_confirmed_interval():
    """THE DEFECT. 114 of 1,601 vs 0 of 22, the recruitment sample's
    race_ethnicity pair. The bootstrap read 5.8 to 8.4 points and confirmed it;
    the Wilson interval for 0 of 22 alone runs to 14.9%, so the gap cannot be
    established from 22 people."""
    df, y = _two_groups(("White", 1601, 114), ("Native American", 22, 0))
    row = o._per_variable(df, "g", y, None)

    assert row["referenceGroup"] == "White" and row["worstGroup"] == "Native American"
    assert row["ciLow"] < 0.0 < row["ciHigh"], (row["ciLow"], row["ciHigh"])
    assert row["significantRaw"] is False and row["significant"] is False
    assert row["pValue"] > 0.05
    assert row["ciLow"] == pytest.approx(-0.077901, abs=1e-5)
    assert row["ciHigh"] == pytest.approx(0.084856, abs=1e-5)
    assert row["ciMethod"] == "newcombe_hybrid_score"
    assert row["ciLevel"] == pytest.approx(0.95)
    # A zero-rate group is still exclusion on the four-fifths screen, but an
    # unconfirmed result is at most a watch item.
    assert row["tone"] != "critical"


def test_a_confirmed_exclusion_with_enough_people_stays_confirmed():
    """The recruitment sample's age pair: 30 of 646 vs 0 of 318. The worst
    group has no positives but 318 people, enough to establish the gap."""
    df, y = _two_groups(("45-54", 646, 30), ("18-24", 318, 0))
    row = o._per_variable(df, "g", y, None)

    assert row["ciLow"] == pytest.approx(0.028255, abs=1e-5)
    assert row["ciHigh"] == pytest.approx(0.065521, abs=1e-5)
    assert row["ciMethod"] == "newcombe_hybrid_score"
    assert row["pValue"] < 0.001
    assert row["significantRaw"] is True
    assert row["tone"] == "critical"


def test_a_healthy_count_gap_barely_moves():
    """CONTROL FOR OVER-CORRECTION, pinned to the new method. At 300 of 1,000
    vs 200 of 1,000 the Newcombe interval sits within 0.001 of the Wald
    interval (and of what the bootstrap used to give), and stays confirmed."""
    df, y = _two_groups(("A", 1000, 300), ("B", 1000, 200))
    row = o._per_variable(df, "g", y, None)

    w_lo, w_hi = _wald(300, 1000, 200, 1000)
    assert abs(row["ciLow"] - w_lo) < 1e-3 and abs(row["ciHigh"] - w_hi) < 1e-3
    assert row["ciLow"] == pytest.approx(0.062143, abs=1e-5)
    assert row["ciHigh"] == pytest.approx(0.137500, abs=1e-5)
    assert row["ciMethod"] == "newcombe_hybrid_score"
    assert row["significantRaw"] is True and row["pValue"] < 1e-5


@pytest.mark.parametrize(
    "decisions,reason",
    [
        pytest.param("tiny", "Fewer than 5 rows", id="worst-group-under-5"),
        pytest.param("scores", "not 0/1", id="decisions-not-binary"),
    ],
)
def test_no_interval_is_the_third_state_never_a_number(decisions, reason):
    if decisions == "tiny":
        df, y = _two_groups(("A", 200, 60), ("B", 4, 0))
        y = y.astype(float)
    else:
        df, y = _two_groups(("A", 200, 60), ("B", 200, 30))
        y = np.where(y == 1, 0.9, 0.2)
    row = o._per_variable(df, "g", y, None, min_group=2)

    assert row["ciLow"] is None and row["ciHigh"] is None
    assert row["pValue"] is None and row["significant"] is False
    assert row["ciMethod"] is None and row["ciLevel"] is None
    assert reason in row["ciNotComputedReason"]


# ------------------------- 1c. the reported pair is the most extreme of several


def test_the_worst_pair_is_corrected_for_being_chosen_from_several():
    """THE CLEAN-CONTROL GUARD, at unit level. The worst group is the lowest-rate
    of k-1 groups compared with the reference, so its p is the smallest of k-1.
    32 of 150 against 600 of 2,000: the pair's 95% interval excludes zero
    (p 0.025) and the omnibus test across the groups rejects (p 0.004), so
    only the choice can stop it: picked from 3 comparisons it is p 0.075 and
    not confirmed. Without this, the published clean control (nothing planted)
    confirmed age at BH 0.034 once the interval stopped hiding behind bootstrap
    resolution."""
    df, y = _groups(("A", 2000, 600), ("B", 200, 70), ("C", 200, 44), ("D", 150, 32))
    row = o._per_variable(df, "g", y, None)

    assert row["referenceGroup"] == "A" and row["worstGroup"] == "D"
    assert row["pValueOmnibus"] < 0.05, row["pValueOmnibus"]
    assert row["ciLow"] > 0.0 and row["pValue"] < 0.05, row
    assert row["significantRaw"] is False and row["significant"] is False
    assert row["selectionComparisons"] == 3
    assert row["pValueSelection"] == pytest.approx(3 * row["pValue"])
    assert row["pValueSelection"] >= 0.05


def test_the_same_pair_on_its_own_is_confirmed():
    """CONTROL: with no other group to choose from there is nothing to correct."""
    df, y = _two_groups(("A", 2000, 600), ("D", 150, 32))
    row = o._per_variable(df, "g", y, None)

    assert row["selectionComparisons"] == 1
    assert row["pValueSelection"] == pytest.approx(row["pValue"])
    assert row["significantRaw"] is True


def test_when_both_ends_are_chosen_every_pair_counts():
    """The majority is the worst-served group, so the reference becomes the
    best-served one: the pair is the most extreme of all k(k-1)/2 pairs."""
    df, y = _groups(("A", 1000, 100), ("B", 200, 40), ("C", 200, 30))
    row = o._per_variable(df, "g", y, None)

    assert row["worstGroup"] == "A" and row["referenceGroup"] == "B"
    assert row["selectionComparisons"] == 3
    assert row["pValueSelection"] == pytest.approx(min(1.0, 3 * row["pValue"]))


def test_a_requested_reference_counts_the_groups_compared_with_it():
    df, y = _groups(("A", 1000, 300), ("B", 200, 40), ("C", 200, 30), ("D", 200, 50))
    row = o._per_variable(df, "g", y, None, reference_override="B")

    assert row["referenceOverrideHonored"] is True and row["referenceGroup"] == "B"
    assert row["selectionComparisons"] == 3
    assert row["pValueSelection"] == pytest.approx(min(1.0, 3 * row["pValue"]))


# ------------------------------------- 2. a graded Fail must raise a finding


def _age_row(**overrides) -> dict:
    """The recruitment sample's age row as run_pulse emitted it on 2026-10-09
    (tone critical, confirmed, gap 4.6 points, ratio 0.00)."""
    row = {
        "assessable": True,
        "attribute": "age",
        "worstGroup": "18-24",
        "referenceGroup": "45-54",
        "gap": 0.0464,
        "ciLow": 0.0283,
        "ciHigh": 0.0655,
        "fourFifthsRatio": 0.0,
        "significant": True,
        "significantRaw": True,
        "tone": "critical",
    }
    row.update(overrides)
    return row


def _dsp(verdict: dict, attribute: str) -> list:
    return [
        f
        for f in verdict["findings"]
        if f["type"] == "disparate_impact" and f.get("attribute") == attribute
    ]


def test_a_confirmed_fail_below_five_points_raises_a_finding_and_a_recommendation():
    """THE DEFECT. Graded Fail, confirmed, 4.6 points: no DSP finding before."""
    v = build_assurance_verdict(per_variable=[_age_row()], domain="hiring", jurisdiction="EU")
    hits = _dsp(v, "age")
    assert len(hits) == 1, v["findings"]
    f = hits[0]
    # The severity the status already states: critical is what "Fail" means.
    assert f["severity"] == "critical"
    assert "Graded Fail" in f["evidence"]
    assert "four-fifths ratio 0.00" in f["evidence"]
    assert "95% CI 3 to 7 pts" in f["evidence"]
    recs = [r for r in v["recommendations"] if f["id"] in r["addressesFindings"]]
    assert recs, "a Fail must be traceable to a recommendation"
    assert v["overall"] == "Adverse"


def test_every_graded_fail_is_traceable_to_a_finding():
    """The rule as a property: over every combination of gap size, tone,
    confirmation and jurisdiction, a confirmed critical row always has a DSP
    finding at critical and a recommendation behind it."""
    for gap in (0.005, 0.02, 0.0464, 0.049, 0.06, 0.15, 0.3):
        for tone in ("pass", "warn", "critical"):
            for sig in (True, False):
                for jur in ("EU", "US", "CH"):
                    row = _age_row(gap=gap, tone=tone, significant=sig)
                    v = build_assurance_verdict(
                        per_variable=[row], domain="hiring", jurisdiction=jur
                    )
                    hits = _dsp(v, "age")
                    if tone == "critical" and sig:
                        assert len(hits) == 1, (gap, tone, sig, jur)
                        assert hits[0]["severity"] == "critical", (gap, tone, sig, jur)
                        assert any(
                            hits[0]["id"] in r["addressesFindings"] for r in v["recommendations"]
                        )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"significant": False, "tone": "warn"}, id="unconfirmed-capped-warn"),
        pytest.param({"significant": False, "tone": "critical"}, id="unconfirmed-critical"),
        pytest.param({"significant": None, "tone": "critical"}, id="confirmation-unknown"),
        pytest.param({"tone": "warn"}, id="confirmed-but-graded-warn"),
        pytest.param({"tone": None}, id="no-status"),
    ],
)
def test_below_five_points_without_a_confirmed_fail_still_raises_nothing(overrides):
    """CONTROL (passes before and after): the new rule follows the status and
    its confirmation, it does not lower the 5-point line for everything."""
    v = build_assurance_verdict(
        per_variable=[_age_row(**overrides)], domain="hiring", jurisdiction="EU"
    )
    assert _dsp(v, "age") == []


def test_the_interval_prints_with_to_so_a_negative_bound_reads_clearly():
    v = build_assurance_verdict(
        per_variable=[
            _age_row(
                attribute="race_ethnicity",
                worstGroup="Native American",
                referenceGroup="White",
                gap=0.0712,
                ciLow=-0.0779,
                ciHigh=0.0849,
                significant=False,
                tone="warn",
            )
        ],
        domain="hiring",
        jurisdiction="EU",
    )
    (f,) = _dsp(v, "race_ethnicity")
    assert "95% CI -8 to 8 pts" in f["evidence"]
    assert "not yet significant" in f["evidence"]
    assert f["severity"] == "warn"


# ------------------- 4. an empty or under-floor grid cell is not measured


def _grid_frame():
    """A x/y measured (40 rows each), B x has 3 rows, B y is empty."""
    rows = (
        [("A", "x", 1)] * 20
        + [("A", "x", 0)] * 20
        + [("A", "y", 1)] * 8
        + [("A", "y", 0)] * 32
        + [("B", "x", 0)] * 3
    )
    df = pd.DataFrame(rows, columns=["g", "s", "y"])
    return df, df["y"].to_numpy()


def test_an_empty_or_under_floor_grid_cell_carries_no_gap():
    """THE DEFECT (quarantined pipeline.py, same class as the platform heatmap). An
    empty cell was given a gap of 0.0 and no size floor applied, so a 3-person
    cell at 0% became the worst cell (gap 0.5) and an empty one read as parity.
    Now both carry no gap, keep their n, and stay out of the worst cell and the
    concentration."""
    from vfairness.operations.pulse.pipeline import _intersectional_grid

    df, y = _grid_frame()
    rows, cols, cells, concentration = _intersectional_grid(df, "g", "s", y, 0.5)
    by = {(c["rowLabel"], c["colLabel"]): c for c in cells}

    assert by[("B", "y")]["gap"] is None and by[("B", "y")]["n"] == 0
    assert by[("B", "x")]["gap"] is None and by[("B", "x")]["n"] == 3
    assert by[("B", "x")]["measured"] is False and by[("B", "y")]["measured"] is False
    assert not by[("B", "x")]["worst"] and not by[("B", "y")]["worst"]
    assert by[("A", "y")]["worst"] and by[("A", "y")]["gap"] == pytest.approx(0.3)
    assert by[("A", "x")]["gap"] == pytest.approx(0.0) and by[("A", "x")]["measured"] is True
    assert concentration == pytest.approx(1.0)


def test_a_grid_with_no_measured_cell_has_no_concentration():
    from vfairness.operations.pulse.pipeline import _intersectional_grid

    df, y = _grid_frame()
    _rows, _cols, cells, concentration = _intersectional_grid(df, "g", "s", y, 0.5, min_group=100)
    assert concentration is None
    assert all(c["gap"] is None and not c["worst"] for c in cells)
    # The single-row form applies the same floor.
    _rows, _cols, cells1, _conc = _intersectional_grid(df, "g", None, y, 0.5)
    by = {c["rowLabel"]: c for c in cells1}
    assert by["B"]["gap"] is None and by["B"]["n"] == 3
    assert by["A"]["measured"] is True and by["A"]["worst"] is True


# ------------------------------------------------- both, end to end in run_pulse


def _recruitment_like_frame() -> pd.DataFrame:
    """Two attributes shaped like the recruitment sample, at a low base rate.

    ethnicity: White 21/700, Asian 10/300, Black 1/300, Native American 0/22.
      The across-groups test sees a real difference (Black), the selected pair
      is White vs Native American, and 22 people cannot establish that gap.
    age_band: 18-24 has 0 of 300 selected, the other two bands about 3%, so the
      exclusion is confirmed while the gap to the reference stays under 5 points.
    """
    rng = np.random.default_rng(7)
    rows = []
    for label, n, k in (
        ("White", 700, 21),
        ("Asian", 300, 10),
        ("Black", 300, 1),
        ("Native American", 22, 0),
    ):
        d = np.array([1] * k + [0] * (n - k))
        rng.shuffle(d)
        rows += [(label, int(x)) for x in d]
    df = pd.DataFrame(rows, columns=["ethnicity", "decision"])
    df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    n = len(df)
    age = np.array(["25-44"] * n, dtype=object)
    young = rng.choice(np.flatnonzero(df["decision"].to_numpy() == 0), 300, replace=False)
    age[young] = "18-24"
    rest = np.setdiff1d(np.arange(n), young)
    age[rng.choice(rest, len(rest) // 2, replace=False)] = "45-64"
    df["age_band"] = age
    df["experience"] = rng.normal(size=n)
    return df


@pytest.mark.slow
def test_run_pulse_confirms_only_what_its_interval_and_correction_support():
    df = _recruitment_like_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = o.run_pulse(
            df,
            {
                "source_kind": "predictive",
                "domain": "hiring",
                "jurisdiction": "EU",
                "protected_attributes": ["ethnicity", "age_band"],
            },
        )["data"]
    rows = {r["attribute"]: r for r in res["perVariable"] if r.get("assessable")}

    eth = rows["ethnicity"]
    assert eth["worstGroup"] == "Native American" and eth["referenceGroup"] == "White"
    assert eth["ciLow"] < 0.0 < eth["ciHigh"], eth
    assert eth["significant"] is False and eth["tone"] != "critical", eth

    age = rows["age_band"]
    assert age["worstGroup"] == "18-24"
    assert age["gap"] < 0.05, age
    assert age["significant"] is True and age["tone"] == "critical", age

    # One decision: confirmed implies the interval excludes zero, the pair
    # survives its selection correction, the pre-correction verdict held, and
    # the BH-adjusted p clears alpha.
    for r in rows.values():
        if r["significant"]:
            assert r["significantRaw"] and r["ciLow"] > 0.0, r
            assert r["pValueSelection"] < 0.05, r
            assert r["pValueFamily"] == pytest.approx(r["pValueSelection"]), r
            assert r["pValueAdjusted"] <= 0.05, r

    # Every attribute graded Fail has a finding and a recommendation.
    findings = res["assurance"]["findings"]
    recs = res["assurance"]["recommendations"]
    for attr, r in rows.items():
        if r["tone"] == "critical":
            hits = [
                f
                for f in findings
                if f["type"] == "disparate_impact" and f.get("attribute") == attr
            ]
            assert len(hits) == 1 and hits[0]["severity"] == "critical", (attr, findings)
            assert any(hits[0]["id"] in rec["addressesFindings"] for rec in recs), attr
