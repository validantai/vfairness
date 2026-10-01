"""Row-level fabrication guards for the two sites that went unassigned.

Both charts scanned a collection of rows and then reported a RESULT for every
row, whether or not the row reported anything:

* ``reweighting_comparison_to_svg`` reached every cell through a
  ``.get(key, 0)`` default, so a method result that carried nothing but its own
  name was written onto the canvas as four measurements: ``+0.0%`` fairness
  improvement, ``+0.00%`` accuracy change (emerald, because ``0 >= 0`` takes the
  improved arm), ``0.000`` calibration change (emerald, because ``0 <= 0`` takes
  the did-not-get-worse arm) and a ``0.000`` trade-off score. It also took a
  place in the subtitle count, so four zeros were presented as four comparisons.

* ``intersectional_disparity_to_svg`` gave an unmeasured subgroup a 0.0%
  positive rate, a 0.0pp delta, ``n=0``, a severity defaulting to ``"low"`` and
  a place in "7 subgroups analysed"; the same 0.0% was then used as the SORT
  KEY, so a subgroup nobody measured was ranked at the bottom of a ranking that
  is read as most-disadvantaged-last. The chart-level ``disparitySeverity``
  defaulted to ``"medium"``, which bands a gap that nobody graded.

The rule these tests encode: a row that reported nothing gets no number, no
badge, no colour, no plot point, and no place in any count or headline that
implies it was measured. A default is not a measurement; a sentinel is not a
measurement; absent and zero are different claims. The headline is graded over
the graded subset, never reads as an unqualified all-clear while anything is
ungraded, states the ungraded count ON THE CANVAS in the headline band
(y <= 200), and never lets an ungraded row into a numerator or a denominator.

The CONTROL tests at the bottom are as load-bearing as the rest: a genuine 0.0
IS a measurement and must still be drawn, and a fully reported run must render
exactly as it did before this rule existed.
"""

import re

import pytest

from vfairness.rendering.adapters_feature_engineering import intersectional_disparity_to_svg
from vfairness.rendering.adapters_post_processing import reweighting_comparison_to_svg
from vfairness.rendering.skins import BLANCO_MAP

# The band a reader takes the verdict from. Part (c) of the headline rule puts
# the ungraded count here, not at the foot of a tall canvas the reader may never
# scroll to.
HEADLINE_BAND_Y = 200


def _texts(svg):
    """[(y, text), ...] for every <text> element on the canvas."""
    out = []
    for match in re.finditer(r"<text[^>]*\by=\"([-\d.]+)\"[^>]*>(.*?)</text>", svg, re.S):
        out.append((float(match.group(1)), re.sub(r"<[^>]+>", "", match.group(2))))
    return out


def _canvas(svg):
    """The drawn canvas only, with the accessible <desc>/<metadata> stripped.

    The description is asserted separately; folding the two together lets a
    phrase present in one stand in for the other.
    """
    body = re.sub(r"<desc>.*?</desc>", "", svg, flags=re.S)
    return re.sub(r"<metadata.*?</metadata>", "", body, flags=re.S)


def _desc(svg):
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    return match.group(1) if match else ""


def _in_headline_band(svg, needle):
    """True when *needle* is drawn at y <= 200."""
    return any(needle in text and y <= HEADLINE_BAND_Y for y, text in _texts(svg))


def _skinned(hex_colour):
    """The hex a source colour is REWRITTEN to before it reaches the canvas.

    ``render_svg`` applies the Blanco skin, so asserting on the palette hex an
    adapter writes (``#d97706``) matches nothing at all and passes for the wrong
    reason. The map is read here rather than transcribed, so a palette change
    cannot silently make these assertions vacuous.
    """
    return BLANCO_MAP.get(hex_colour, hex_colour)


def _bars_region(canvas):
    """The ranked-bar block only, so the comparison cards cannot answer for it."""
    return canvas.split("All Subgroups", 1)[1].split("Ground truth", 1)[0]


# ── fixtures ────────────────────────────────────────────────────────────────

GRADED_METHOD = {
    "method": "group_specific",
    "original_fairness": {"demographic_parity_diff": 0.18},
    "adjusted_fairness": {"demographic_parity_diff": 0.04},
    "original_performance": {"accuracy": 0.85},
    "adjusted_performance": {"accuracy": 0.82},
    "calibration_metrics": {"ece_change": 0.02},
    "trade_off_score": 0.85,
}

# A method result that carries nothing but its own name. This is the row the
# whole file is about.
SILENT_METHOD = {"method": "never_ran"}

GRADED_SUBGROUP = {
    "group": "Asian Male",
    "positiveRate": 0.71,
    "groundTruthRate": 0.68,
    "predictionDelta": 0.03,
    "size": 150,
    "severity": "low",
}

SILENT_SUBGROUP = {"group": "Hispanic Nonbinary"}


def _intersectional(groups, **overrides):
    payload = {
        "privilegedGroup": {
            "group": "Asian Male",
            "positiveRate": 0.71,
            "groundTruthRate": 0.68,
            "size": 150,
        },
        "disadvantagedGroup": {
            "group": "Black Female",
            "positiveRate": 0.34,
            "groundTruthRate": 0.40,
            "size": 195,
        },
        "allGroups": groups,
        "maxDisparity": 0.37,
        "disparitySeverity": "high",
        "insights": ["Black Female shows a 37pp approval gap."],
        "findings": [{"severity": "high"}],
    }
    payload.update(overrides)
    return payload


# ── 1. reweighting: a silent row gets no number ─────────────────────────────


def test_reweighting_silent_method_gets_no_fabricated_numbers():
    svg = _canvas(
        reweighting_comparison_to_svg(
            {"method_results": [GRADED_METHOD, SILENT_METHOD], "best_method": "group_specific"}
        )
    )
    assert "never_ran" in svg, "the row must still be listed, so the reader sees it was named"
    row = svg[svg.index("never_ran") :]
    for fabricated in ("+0.0%", "+0.00%", "0.000"):
        assert fabricated not in row, (
            f"the silent method row printed {fabricated!r}, which is this adapter's own "
            f"default presented as a measurement"
        )
    assert "NOT MEASURED" in row


def test_reweighting_silent_method_is_outside_the_subtitle_count():
    svg = _canvas(
        reweighting_comparison_to_svg(
            {"method_results": [GRADED_METHOD, SILENT_METHOD], "best_method": "group_specific"}
        )
    )
    assert "1 methods compared" in svg, "the count is over the GRADED rows only"
    assert "2 methods compared" not in svg, (
        "a row that reported nothing was counted as a method compared"
    )


def test_reweighting_silent_method_is_outside_the_description_denominator():
    desc = _desc(
        reweighting_comparison_to_svg(
            {"method_results": [GRADED_METHOD, SILENT_METHOD], "best_method": "group_specific"}
        )
    )
    assert "across 1 methods" in desc
    assert "across 2 methods" not in desc, (
        "the accessible description counted an ungraded row in its denominator"
    )


def test_reweighting_ungraded_count_is_stated_in_the_headline_band():
    svg = reweighting_comparison_to_svg(
        {"method_results": [GRADED_METHOD, SILENT_METHOD], "best_method": "group_specific"}
    )
    assert _in_headline_band(svg, "not graded"), (
        "part (c): the ungraded count must be on the canvas in the headline band "
        "(y <= 200), where the reader takes the verdict from, not at the foot"
    )


def test_reweighting_partially_reported_method_keeps_the_cell_it_did_report():
    """Withholding is per CELL as well as per row; it is not all-or-nothing."""
    partial = {
        "method": "partial_only",
        "original_performance": {"accuracy": 0.85},
        "adjusted_performance": {"accuracy": 0.88},
    }
    svg = _canvas(
        reweighting_comparison_to_svg(
            {"method_results": [GRADED_METHOD, partial], "best_method": "group_specific"}
        )
    )
    row = svg[svg.index("partial_only") :]
    assert "+3.00%" in row, "the accuracy change WAS reported and must still print"
    assert row.count("not measured") == 3, (
        "fairness, calibration and trade-off reported nothing and must each say so"
    )
    assert "2 methods compared" in svg, "a row with one measured cell IS a compared method"


def test_reweighting_every_row_silent_is_could_not_check():
    svg = reweighting_comparison_to_svg({"method_results": [SILENT_METHOD, {"method": "b"}]})
    assert "COULD NOT CHECK" in _canvas(svg).upper()
    assert "COULD NOT CHECK" in _desc(svg).upper()
    assert not re.search(r"\d+ methods compared", _canvas(svg)), (
        "a table of nothing but unmeasured rows may not carry a comparison headline"
    )


def test_reweighting_best_method_without_a_graded_row_is_not_painted_as_a_recommendation():
    """The emerald stripe and emerald name ARE the recommendation."""
    svg = _canvas(
        reweighting_comparison_to_svg(
            {"method_results": [GRADED_METHOD, SILENT_METHOD], "best_method": "never_ran"}
        )
    )
    badge = svg[svg.index("BEST METHOD") : svg.index("BEST METHOD") + 400]
    assert "not graded here" in badge
    assert "#059669" not in badge, (
        "a method with nothing measured behind it was styled as the recommended one"
    )


def test_reweighting_silent_row_does_not_rescale_the_tradeoff_bars():
    """An ungraded row may not enter the maximum every other bar is scaled to."""

    def bar_width(svg):
        match = re.search(r'<rect x="576"[^>]*width="([\d.]+)"', _canvas(svg))
        return match.group(1) if match else None

    alone = reweighting_comparison_to_svg({"method_results": [GRADED_METHOD]})
    with_silent = reweighting_comparison_to_svg({"method_results": [GRADED_METHOD, SILENT_METHOD]})
    assert bar_width(alone) is not None
    assert bar_width(alone) == bar_width(with_silent)


# ── 2. intersectional: a silent subgroup gets no number and no severity ─────


def test_intersectional_silent_subgroup_gets_no_rate_no_delta_and_no_n():
    svg = _canvas(
        intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, SILENT_SUBGROUP]))
    )
    assert "Hispanic Nonbinary" in svg, "the subgroup must still be listed"
    row = svg[svg.index("Hispanic Nonbinary") :]
    assert "0.0%" not in row, "an unmeasured subgroup was given a measured 0.0% positive rate"
    assert "0.0pp" not in row, "an unmeasured subgroup was given a measured 0.0pp delta"
    assert "n=0<" not in row, "an unmeasured subgroup was given a sample size of 0"
    assert "NOT MEASURED" in row


def test_intersectional_silent_subgroup_is_outside_the_subtitle_count():
    svg = _canvas(
        intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, SILENT_SUBGROUP]))
    )
    assert "1 subgroups analysed" in svg
    assert "2 subgroups analysed" not in svg, (
        "a subgroup that reported no positive rate was counted as analysed"
    )
    assert "1 groups ranked by positive rate" in svg
    assert "2 groups ranked by positive rate" not in svg


def test_intersectional_ungraded_count_is_stated_in_the_headline_band():
    svg = intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, SILENT_SUBGROUP]))
    assert _in_headline_band(svg, "not graded"), (
        "part (c): the ungraded count belongs in the headline band (y <= 200)"
    )


def test_intersectional_silent_subgroup_is_not_sorted_by_a_fabricated_zero():
    """The 0.0% default was also the sort key, so it RANKED an absence."""
    svg = _canvas(
        intersectional_disparity_to_svg(
            _intersectional(
                [
                    SILENT_SUBGROUP,
                    dict(GRADED_SUBGROUP, group="Low Rate Group", positiveRate=0.02),
                    GRADED_SUBGROUP,
                ]
            )
        )
    )
    bars = _bars_region(svg)
    ranked = [t for _, t in _texts(bars) if t in ("Asian Male", "Low Rate Group")]
    assert ranked == ["Asian Male", "Low Rate Group"], "graded rows rank by their own rate"
    # The rank column decides it. A fabricated 0.0% does not merely place the
    # row last, it GIVES IT A RANK, and a rank on this chart is read as a
    # position in a measured ordering. Two graded rows, then an unranked one.
    rank_labels = re.findall(r'<text x="44"[^>]*text-anchor="end">([^<]*)</text>', bars)
    assert rank_labels == ["1", "2", "-"], (
        f"the rank column reads {rank_labels}: an unmeasured subgroup was given a "
        f"rank in a ranking it was never in"
    )
    assert bars.index("Low Rate Group") < bars.index("Hispanic Nonbinary"), (
        "an unmeasured subgroup was ordered below a genuinely low-rate one, which "
        "reads as the worst-affected group in the chart"
    )


def test_intersectional_per_group_severity_is_never_defaulted():
    """A severity is a judgement; defaulting one manufactures a finding."""
    no_severity = {
        "group": "Black Female",
        "positiveRate": 0.34,
        "groundTruthRate": 0.40,
        "predictionDelta": -0.06,
        "size": 195,
    }
    bars = _bars_region(
        _canvas(intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, no_severity])))
    )
    # One chunk per ranked row. cx=56 is the row's severity dot and it is drawn
    # BEFORE the group name, so slicing from the name would lose it; the legend
    # circles further down are cx=44 / cx=134 and cannot answer for it either.
    rows = re.split(r'(?=<circle cx="56")', bars)
    row = next(chunk for chunk in rows if "Black Female" in chunk)
    dot = re.search(r'<circle cx="56"[^>]*>', row)
    assert dot is not None and 'fill="none"' in dot.group(0), (
        "a subgroup with no reported severity was dotted like a graded one; the "
        "absence of a severity must not be drawn as a point on the severity scale"
    )
    # The LOW bar colour. The old default painted this row as a graded
    # low-severity finding, which is an all-clear nobody made.
    bar = re.search(r'<rect x="220"[^>]*fill="(#[0-9a-f]{6})"', row)
    assert bar is None or bar.group(1) != _skinned("#64748b"), (
        "a subgroup with no reported severity was painted in the LOW severity colour"
    )


def test_intersectional_graded_row_withholds_only_the_cells_it_did_not_report():
    """Withholding is per CELL here too: a measured rate does not license the rest.

    ``predictionDelta`` defaulted to 0, which says the prediction and the ground
    truth agree exactly, and ``size`` defaulted to 0, which is a claim about how
    many people are in the subgroup.
    """
    rate_only = {"group": "Black Female", "positiveRate": 0.34, "severity": "high"}
    bars = _bars_region(
        _canvas(intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, rate_only])))
    )
    row = bars[bars.index("Black Female") :]
    assert "34.0%" in row, "the rate WAS reported and must still print"
    assert "0.0pp" not in row, "an unreported delta was drawn as a measured 0.0pp"
    assert "n=0<" not in row, "an unreported subgroup size was drawn as a measured 0"
    assert "2 subgroups analysed" in _canvas(
        intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, rate_only]))
    ), "a subgroup with a measured rate IS analysed"


def test_intersectional_chart_severity_is_never_defaulted_to_medium():
    payload = _intersectional([GRADED_SUBGROUP])
    del payload["disparitySeverity"]
    svg = _canvas(intersectional_disparity_to_svg(payload))
    assert "MEDIUM" not in svg, (
        "the disparity band defaulted to MEDIUM, which grades a gap nobody graded"
    )
    assert "NOT GRADED" in svg
    assert _skinned("#d97706") not in svg, "the canvas was painted in the MEDIUM severity colour"


def test_intersectional_missing_ground_truth_draws_no_ground_truth_bar():
    """Defaulting ground truth to the PREDICTED rate claims perfect agreement."""
    no_gt = {"group": "Black Female", "positiveRate": 0.34, "size": 195, "severity": "high"}
    bars = _bars_region(_canvas(intersectional_disparity_to_svg(_intersectional([no_gt]))))
    # Both bars start at x=220; the ground-truth one is the pale fill behind.
    fills = re.findall(r'<rect x="220"[^>]*fill="(#[0-9a-f]{6})"', bars)
    assert fills, "the prediction bar must still be drawn"
    assert _skinned("#e2e8f0") not in fills, (
        "a ground-truth bar was drawn from a ground truth nobody reported"
    )


def test_intersectional_card_without_a_rate_says_na_rather_than_zero():
    payload = _intersectional([GRADED_SUBGROUP])
    payload["disadvantagedGroup"] = {"group": "Nonbinary Applicant"}
    svg = _canvas(intersectional_disparity_to_svg(payload))
    card = svg[svg.index("MOST DISADVANTAGED") :]
    card = card[: card.index("All Subgroups")]
    assert "N/A" in card
    assert "0.0%" not in card, "a named subgroup with no rate was given a measured 0.0%"
    assert "n=0<" not in card, "a named subgroup with no size was given n=0"


# ── 3. controls: a real zero is a measurement, and healthy input is unchanged ─


def test_a_genuine_zero_rate_still_ranks_and_still_prints():
    """Over-correction guard. Absent is withheld; a measured 0.0 is not."""
    real_zero = {
        "group": "Zero Rate Group",
        "positiveRate": 0.0,
        "groundTruthRate": 0.0,
        "predictionDelta": 0.0,
        "size": 42,
        "severity": "high",
    }
    svg = _canvas(intersectional_disparity_to_svg(_intersectional([GRADED_SUBGROUP, real_zero])))
    row = svg[svg.index("Zero Rate Group") :]
    assert "0.0%" in row and "0.0pp" in row and "n=42" in row
    assert "NOT MEASURED" not in row
    assert "2 subgroups analysed" in svg


def test_a_genuine_zero_reweighting_cell_still_prints():
    """Over-correction guard on the other chart."""
    real_zeros = {
        "method": "no_op",
        "original_fairness": {"demographic_parity_diff": 0.18},
        "adjusted_fairness": {"demographic_parity_diff": 0.18},
        "original_performance": {"accuracy": 0.85},
        "adjusted_performance": {"accuracy": 0.85},
        "calibration_metrics": {"ece_change": 0.0},
        "trade_off_score": 0.0,
    }
    svg = _canvas(reweighting_comparison_to_svg({"method_results": [real_zeros]}))
    assert "+0.0%" in svg, "a measured zero improvement IS a measurement"
    assert "+0.00%" in svg
    assert "0.000" in svg
    assert "not measured" not in svg
    assert "1 methods compared" in svg


@pytest.mark.parametrize(
    "render, expected",
    [
        (
            lambda: reweighting_comparison_to_svg(
                {
                    "method_results": [GRADED_METHOD],
                    "best_method": "group_specific",
                    "metadata": {"n_samples": 4820, "n_groups": 4},
                }
            ),
            ["+77.8%", "-3.00%", "+0.020", "0.850", "1 methods compared", "4820"],
        ),
        (
            lambda: intersectional_disparity_to_svg(
                _intersectional(
                    [
                        GRADED_SUBGROUP,
                        {
                            "group": "Black Female",
                            "positiveRate": 0.34,
                            "groundTruthRate": 0.40,
                            "predictionDelta": -0.06,
                            "size": 195,
                            "severity": "high",
                        },
                    ]
                )
            ),
            ["71.0%", "34.0%", "3.0pp", "-6.0pp", "n=150", "HIGH", "2 subgroups analysed"],
        ),
    ],
    ids=["reweighting", "intersectional"],
)
def test_a_fully_reported_run_still_draws_every_number(render, expected):
    """The over-correction control. Nothing is withheld when nothing is absent."""
    svg = _canvas(render())
    for token in expected:
        assert token in svg, f"a fully reported run stopped drawing {token!r}"
    for withheld in ("NOT MEASURED", "not measured", "NOT GRADED", "not graded"):
        assert withheld not in svg, (
            f"a fully reported run wrote {withheld!r}, so the guard is over-firing"
        )
