"""LF-20: the disparity that is present in every prompt and visible in none.

The Pulse probe tested one (template, axis) cell at a time and corrected the
cells against each other. That design answers "is there a disparity in THIS
prompt", and it is blind by construction to the commonest real shape: a small
consistent shift applied to every prompt. The spread between different prompts
is far larger than the shift, so no cell reaches significance, and the reader is
told the model treats the arms alike.

Pairing the templates removes that spread, which is what a matched design is
for. These tests measure the claim rather than asserting it: the same synthetic
data is run through both readings and the two answers are compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.operations.pulse.llm_probe import (
    _cell_arm_mean,
    _cell_stats,
    _sentiment_trend_stats,
    apply_bh,
    correct_families,
)


def _cells(deltas: list[float], baselines: list[float], runs: int = 4) -> list[dict]:
    """One cell per template, two arms, a fixed shift applied to the second.

    Each template has its OWN baseline, which is what a real template set looks
    like: "write a recommendation" and "list three concerns" do not produce the
    same sentiment. Within a template the runs vary slightly, so the per-cell
    test has something to work with and is not short-circuited by the
    constant-arms branch.
    """
    rng = np.random.default_rng(4)
    out = []
    for idx, (base, delta) in enumerate(zip(baselines, deltas)):
        ref = list(base + rng.normal(0, 0.02, size=runs))
        arm = list(base + delta + rng.normal(0, 0.02, size=runs))
        arm_scores = [ref, arm]
        p_cell, effect, pairs = _cell_stats(["Reference", "Arm"], arm_scores)
        out.append(
            {
                "axis": "name",
                "templateId": f"t{idx}",
                "armLabels": ["Reference", "Arm"],
                "armMeans": [_cell_arm_mean(ref), _cell_arm_mean(arm)],
                "pValue": p_cell,
                "effectSize": effect,
                "pairs": pairs,
            }
        )
    return out


def test_a_consistent_shift_is_found_by_the_paired_test_and_missed_per_template():
    """The measurement this whole change rests on.

    Twelve templates whose baselines run from -0.5 to +0.5, each shifted by the
    same 0.12. That shift is a quarter of the spread between templates.
    """
    baselines = list(np.linspace(-0.5, 0.5, 12))
    cells = _cells([0.12] * 12, baselines)

    # What the per-template design sees. Its family is the twelve cells, so the
    # rank-1 bar is 0.05/12 = 0.0042.
    per_cell = [c["pValue"] for c in cells if c["pValue"] is not None]
    assert per_cell, "the per-cell test must have run, or this proves nothing"
    assert min(per_cell) > 0.05 / len(cells), (
        "the per-template reading must MISS this, or the comparison below is "
        "not measuring what it claims"
    )

    # What the paired test sees.
    rows = _sentiment_trend_stats(cells)
    assert len(rows) == 1
    row = rows[0]
    assert row["arm"] == "Arm"
    assert row["referenceArm"] == "Reference"
    assert row["pValue"] is not None
    assert row["pValue"] < 0.05 / max(1, len(rows))
    assert row["meanDelta"] == pytest.approx(0.12, abs=0.02)
    assert row["templatesPaired"] == 12
    assert row["templatesDropped"] == 0


def test_no_shift_is_not_found() -> None:
    """Over-correction control. A test that fires on identical arms would be
    worse than the blindness it replaces."""
    baselines = list(np.linspace(-0.5, 0.5, 12))
    rows = _sentiment_trend_stats(_cells([0.0] * 12, baselines))
    assert rows[0]["pValue"] is not None
    assert rows[0]["pValue"] > 0.05
    assert abs(rows[0]["meanDelta"]) < 0.03


def test_a_shift_that_alternates_direction_is_not_a_trend() -> None:
    """Six templates favour the arm and six the reference, by the same amount.

    There IS a disparity in every template and it is not a consistent one, so
    the paired test must not report a trend. The per-template tests are the
    right instrument for that shape, and this one has to stay out of their way.
    """
    baselines = list(np.linspace(-0.5, 0.5, 12))
    deltas = [0.3, -0.3] * 6
    rows = _sentiment_trend_stats(_cells(deltas, baselines))
    assert rows[0]["pValue"] is not None
    assert rows[0]["pValue"] > 0.05


def test_too_few_templates_is_refused_rather_than_reported() -> None:
    """Four templates floor at 0.125 and cannot clear 0.05 under any values."""
    rows = _sentiment_trend_stats(_cells([0.4] * 4, [0.0, 0.2, -0.2, 0.4]))
    row = rows[0]
    assert row["pValue"] is None
    assert row["detectable"] is False
    assert row["minAttainablePValue"] == pytest.approx(0.125)
    assert "could not reach" in row["reason"]


def test_an_unreadable_arm_drops_its_template_and_the_count_says_so() -> None:
    """_cell_arm_mean takes _pair_stats' rule verbatim.

    ANY non-finite score means the arm was not measured in that cell.
    Averaging the readable half would hand the paired test a number built from
    a smaller, content-selected sample than it thinks it has.
    """
    assert _cell_arm_mean([0.1, 0.2]) == pytest.approx(0.15)
    assert _cell_arm_mean([0.1, float("nan")]) is None
    assert _cell_arm_mean([]) is None
    assert _cell_arm_mean([float("inf"), 0.1]) is None

    cells = _cells([0.12] * 12, list(np.linspace(-0.5, 0.5, 12)))
    cells[0]["armMeans"] = [None, cells[0]["armMeans"][1]]
    cells[1]["armMeans"] = [cells[1]["armMeans"][0], None]
    rows = _sentiment_trend_stats(cells)
    assert rows[0]["templatesSupplied"] == 12
    assert rows[0]["templatesPaired"] == 10
    assert rows[0]["templatesDropped"] == 2


def test_a_single_arm_axis_produces_no_comparison() -> None:
    assert _sentiment_trend_stats([]) == []
    lone = _cells([0.1], [0.0])
    lone[0]["armLabels"] = ["Reference"]
    assert _sentiment_trend_stats(lone) == []


# ---------------------------------------------------------------------------
# The family boundary, which is the most consequential decision in the file
# ---------------------------------------------------------------------------


def test_the_trend_family_is_its_own_and_pooling_would_suppress_a_finding():
    """Why _sentiment_trend_stats gets a separate Benjamini-Hochberg family.

    The two questions are different and are answered by different tests: the
    cells ask whether any single prompt is handled unfairly, the trend asks
    whether an arm is handled differently overall. Pooling them spends one
    question's alpha budget on the other, which is exactly what cost the
    refusal probe all of its power before READINESS-5 split it out.

    Measured rather than argued: the same trend p-value is significant in its
    own family of four and is not once the twenty-four sentiment cells are
    poured in beside it.
    """
    trend = [{"arm": f"a{i}", "pValue": p} for i, p in enumerate([0.008, 0.4, 0.6, 0.9])]
    cells = [{"pValue": 0.5 + i * 0.01} for i in range(24)]

    own = [dict(t) for t in trend]
    apply_bh(own)
    assert own[0]["familyWiseSignificant"] is True

    pooled = [dict(t) for t in trend] + [dict(c) for c in cells]
    apply_bh(pooled)
    assert pooled[0]["familyWiseSignificant"] is False, (
        "pooling must be shown to suppress the finding, or this test proves nothing"
    )


def test_an_untested_comparison_stays_out_of_the_family_and_out_of_both_verdicts():
    """A null p was never run. It must not raise the bar for the rest, and it
    must not read as "tested, not significant" either."""
    items = [
        {"arm": "measured", "pValue": 0.02},
        {"arm": "never_ran", "pValue": None},
        {"arm": "never_ran_2", "pValue": None},
    ]
    apply_bh(items)
    assert items[0]["familyWiseSignificant"] is True, (
        "two untested members must not have raised the bar from 0.05 to 0.05/3"
    )
    assert items[1]["familyWiseSignificant"] is None
    assert items[2]["familyWiseSignificant"] is None


def test_the_call_site_keeps_the_three_families_apart():
    """The property the test above could not defend.

    Proving that pooling WOULD suppress a finding says nothing about whether
    the code pools. This calls what the probe calls, with a trend p-value that
    survives its own family of four and dies in a pooled family of
    twenty-eight, and reads the answer off the object the probe reads it off.
    """
    cells = [{"pValue": 0.5 + i * 0.01} for i in range(24)]
    refusal = [{"pValue": 0.9}]
    trend = [{"arm": f"a{i}", "pValue": p} for i, p in enumerate([0.008, 0.4, 0.6, 0.9])]

    correct_families(cells, refusal, trend)

    assert trend[0]["familyWiseSignificant"] is True, (
        "the trend comparison must be corrected against the other trend "
        "comparisons only; pooling it with the 24 cells suppresses it"
    )
    # And the cells were not made easier by the trend rows joining them.
    assert all(c["familyWiseSignificant"] is False for c in cells)
    assert refusal[0]["familyWiseSignificant"] is False
