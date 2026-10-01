"""A ROW that reported nothing gets no number, no badge and no place in a count.

The chart-level rule ("an adapter given the emptiest input it accepts renders
could-not-check") was already met by every chart in
``adapters_feature_engineering``. It was not met PER ROW, so a chart carrying
real data could still invent an individual row out of a ``.get(key, 0)``
default, and the invented row was then read as a measurement:

* ``transformation_comparison_to_svg``: a feature with no AFTER correlation
  defaulted to ``after = 0``, so its reduction came out ``(before - 0) / before
  = 1.0`` and the row printed **100%**, in green, the best result this chart can
  express. It raised AVG REDUCTION, took a place in FEATURES IMPROVED n / n and
  pushed the summary band towards "Effective". The single most flattering number
  on the canvas was produced by the ABSENCE of the measurement it reports.
* ``intersectional_analysis_to_svg``: an intersection the matrix never carried
  defaulted to ``0.00``, took the darkest green on the grid, and became the
  MINIMUM operand of the headline. The canvas read "MAX DISPARITY 0.720 between
  White and ZZ_BARE" in red at accessible severity HIGH. That is a fabricated
  BREACH: it sends a reader after a violation involving a group the run never
  measured, and when they work out why, the tool is what they stop trusting.
  The operands were wrong in a second way too: they were the FIRST and LAST row
  of the grid, which name the true extremes only by coincidence.
* ``proxy_risk_to_svg``: a candidate with ``risk_level=None`` stringified to
  "none", fell through ``_risk_level_style``'s final ``return`` and rendered
  **LOW** on an emerald badge, counted into "LOW: n", with ``r=0.00`` and a
  half-full confidence bar from a 0.5 default. On the one chart whose whole
  purpose is to say whether a feature is a stand-in for a protected attribute, a
  fabricated LOW is the false negative that matters.
* ``proxy_risk.svg`` advanced its badge ``x`` with a ``{% set %}`` INSIDE the
  ``{% for %}``, which does not survive the iteration, so every summary pill was
  drawn at x=36 and the last one painted covered the rest: a scan holding one
  CRITICAL and two LOW findings showed a single emerald "LOW: 2" pill. The
  markup carried "CRITICAL: 1" throughout, which is why a substring assertion
  never caught it and a rasterised read did.

The headline rule these tests hold to, decided for the whole campaign:
a) grade and band over the GRADED subset, so a partially measured run still
   gives a useful verdict on what WAS measured;
b) never render an unqualified all-clear while anything is ungraded;
c) state the ungraded count prominently ON THE CANVAS, beside the headline;
d) an ungraded row never enters a numerator or a denominator that implies it
   was measured.

Every assertion below is on the RENDERED artifact (its drawn text, its badge
geometry, its accessible description), never on an intermediate dict, because
the artifact is what a reader receives. The CONTROLS at the bottom are as
load-bearing as the guards: withdrawing a healthy row's number would be the
mirror-image bug.
"""

import json
import re
from types import SimpleNamespace

from vfairness.rendering.adapters_feature_engineering import (
    intersectional_analysis_to_svg,
    proxy_risk_to_svg,
    transformation_comparison_to_svg,
)

# ── helpers ─────────────────────────────────────────────────────────────────

_TEXT_RE = re.compile(r">([^<>]+)<")
_META_RE = re.compile(
    r'<metadata id="vfairness-explanation"><!\[CDATA\[(.*?)\]\]></metadata>', re.S
)
_A11Y_RE = re.compile(
    r"<title>.*?</title>|<desc>.*?</desc>"
    r'|<metadata id="vfairness-explanation">.*?</metadata>',
    re.S,
)
_BADGE_RE = re.compile(
    r'<text x="(\d+)" y="153"[^>]*>((?:CRITICAL|HIGH|MEDIUM|LOW|NOT RATED): \d+)</text>'
)
# A before/after bar is the only rect on the transformation canvas that is both
# half a row high and 55% opaque; the legend swatches are 10px high. Matching on
# geometry rather than on a colour keeps this independent of the design system's
# palette remapping (it rewrites #dc2626 before the SVG leaves the engine).
_BAR_RE = re.compile(r'<rect [^>]*height="17"[^>]*opacity="0\.55"')


def drawn_text(svg: str) -> str:
    """Everything a sighted reader sees on the canvas, as one string.

    Deliberately NOT the raw markup: the proxy-risk badge bug proves the point,
    because the hidden "CRITICAL: 1" was in the markup the whole time.
    """
    return " ".join(t.strip() for t in _TEXT_RE.findall(_A11Y_RE.sub("", svg)) if t.strip())


def explanation_meta(svg: str) -> dict:
    """The machine-readable explanation block injected by rendering.explain."""
    match = _META_RE.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation metadata block"
    return json.loads(match.group(1))


def badge_boxes(svg: str):
    """(centre_x, label) for each drawn summary pill on the proxy-risk canvas."""
    return [(int(x), label) for x, label in _BADGE_RE.findall(svg)]


def _pv(feature, **kw):
    """A ProxyVariableResult-shaped stand-in. Anything omitted is genuinely absent."""
    return SimpleNamespace(feature=feature, **kw)


_RATED_PROXIES = [
    _pv(
        "zip_code",
        protected_attribute="race",
        correlation=0.72,
        risk_level=SimpleNamespace(value="high"),
        confidence_score=0.91,
    ),
    _pv(
        "income",
        protected_attribute="age",
        correlation=0.45,
        risk_level=SimpleNamespace(value="medium"),
        confidence_score=0.78,
    ),
]


# ── 1. transformation comparison ────────────────────────────────────────────


def test_feature_without_an_after_correlation_is_not_a_100_percent_reduction():
    """THE WORST SURVIVOR. Absence rendered as the best possible result."""
    svg = transformation_comparison_to_svg(
        {"zip_code": 0.72, "income": 0.45, "ZZ_BARE": 0.60},
        {"zip_code": 0.28, "income": 0.15},
    )
    text = drawn_text(svg)

    assert "ZZ_BARE" in text, "the feature is still listed; it was looked at"
    assert "100%" not in text, (
        "a feature with no AFTER measurement rendered a perfect 100% correlation "
        "reduction, the best result this chart can express, from a number nobody took"
    )
    # It is out of the graded counts: two pairs were compared, not three.
    assert "2 / 2" in text, "FEATURES IMPROVED counts the compared pairs only"
    assert "3 / 3" not in text and "2 / 3" not in text
    # The average is over the compared pairs (61% and 66%), not dragged up by a
    # fabricated 100%: (0.611 + 0.666) / 2 = 63%, versus 75% with the invention.
    assert "63%" in text and "75%" not in text
    # Rule (c): the ungraded count is on the canvas, beside the band.
    assert "NOT MEASURED" in text and "1 feature(s), not graded" in text


def test_a_table_where_no_pair_is_complete_cannot_be_banded():
    """Rule (a) has no graded subset to band here, so the whole chart withdraws."""
    svg = transformation_comparison_to_svg({"a": 0.4, "b": 0.5}, {})
    text = drawn_text(svg)
    assert "NOT CHECKED" in text
    for verdict in ("Effective", "Moderate", "Minimal", "AVG REDUCTION"):
        assert verdict not in text, f"{verdict!r} survived a run that compared nothing"
    assert str(explanation_meta(svg)["finding"]).upper().startswith("COULD NOT CHECK")


def test_transformation_row_with_no_after_value_draws_no_bar_and_no_percentage():
    """A zero-length after-bar IS the picture of a perfect transformation."""
    svg = transformation_comparison_to_svg({"zip_code": 0.72, "ZZ_BARE": 0.60}, {"zip_code": 0.28})
    assert "no before/after pair, not compared" in drawn_text(svg)
    # One measured row draws one before-bar and one after-bar; the unmeasured
    # row must add neither, so the pair count follows the MEASURED rows only.
    assert len(_BAR_RE.findall(svg)) == 2
    both = transformation_comparison_to_svg(
        {"zip_code": 0.72, "income": 0.45}, {"zip_code": 0.28, "income": 0.15}
    )
    assert len(_BAR_RE.findall(both)) == 4


def test_a_non_finite_after_correlation_is_absence_not_zero():
    """NaN is not a measurement either, and neither is None."""
    for missing in (float("nan"), None, float("inf")):
        text = drawn_text(
            transformation_comparison_to_svg({"a": 0.5, "b": 0.4}, {"a": 0.2, "b": missing})
        )
        assert "100%" not in text, f"after={missing!r} was read as a measured zero"
        assert "1 / 1" in text, f"after={missing!r} took a place in the graded count"


# ── 2. intersectional analysis ──────────────────────────────────────────────


_PARTIAL_MATRIX = {
    "x_attr": "Gender",
    "y_attr": "Race",
    "matrix": {
        "White": {"Male": 0.65, "Female": 0.72},
        "Black": {"Male": 0.42, "Female": 0.38},
        "ZZ_BARE": {},
    },
    "counts": {"White": {"Male": 320, "Female": 310}, "Black": {"Male": 180, "Female": 195}},
}


def test_an_unmeasured_group_never_becomes_an_operand_of_the_headline():
    """FABRICATED BREACH. The named breach was against a placeholder."""
    svg = intersectional_analysis_to_svg(_PARTIAL_MATRIX, "loan_approval_rate")
    text = drawn_text(svg)

    assert "0.720" not in text, (
        "the headline disparity was measured against cells that defaulted to 0.00"
    )
    assert "0.340" in text, "the spread across the POPULATED cells is 0.72 - 0.38"
    # ZZ_BARE is still on the grid (it was supplied) but names no breach.
    between = text.split("Between", 1)[1]
    assert "ZZ_BARE" not in between, "a group with no measured cell named a breach"
    assert "White · Female" in between and "Black · Female" in between
    # Rule (c).
    assert "NOT MEASURED" in text and "2 of 6 cells, not graded" in text
    assert explanation_meta(svg)["severity"] in ("high", "medium", "low")


def test_an_unpopulated_cell_is_not_drawn_as_a_measured_zero():
    svg = intersectional_analysis_to_svg(_PARTIAL_MATRIX, "loan_approval_rate")
    text = drawn_text(svg)
    assert "0.00" not in text, "an absent intersection was drawn as a rate of exactly zero"
    assert text.count("N/A") == 2 and "not measured" in text
    # Slate, not the emerald that 0.00 took as the best cell on the grid.
    assert "#ecfdf5" not in svg or svg.count('fill="#f1f5f9"') >= 2


def test_the_headline_names_the_cells_that_actually_hold_the_extremes():
    """The operands used to be the first and last ROW of the grid.

    Here the maximum is in row 1 and the minimum in row 2 of three, so naming
    rows 1 and 3 would name a pair that is neither.
    """
    svg = intersectional_analysis_to_svg(
        {
            "x_attr": "G",
            "y_attr": "R",
            "matrix": {
                "r1": {"a": 0.90, "b": 0.80},
                "r2": {"a": 0.10, "b": 0.30},
                "r3": {"a": 0.50, "b": 0.55},
            },
        },
        "rate",
    )
    text = drawn_text(svg)
    assert "0.800" in text, "spread is 0.90 - 0.10"
    between = text.split("Between", 1)[1]
    assert "r1 · a" in between and "r2 · a" in between
    assert "r3" not in between


def test_a_single_populated_cell_is_not_perfect_parity():
    """max - min over ONE cell is 0 by arithmetic, not by measurement."""
    svg = intersectional_analysis_to_svg(
        {"matrix": {"White": {"Male": 0.65}, "Black": {"Male": None}}}, "approval"
    )
    text = drawn_text(svg)
    assert "NOT CHECKED" in text
    assert "MAX DISPARITY" not in text and "0.000" not in text
    assert str(explanation_meta(svg)["finding"]).upper().startswith("COULD NOT CHECK")


# ── 3. proxy risk ───────────────────────────────────────────────────────────


def test_a_candidate_with_no_risk_level_is_not_cleared_as_low_risk():
    """FABRICATED ALL-CLEAR on the chart where a false clear does the damage."""
    svg = proxy_risk_to_svg(
        _RATED_PROXIES
        + [_pv("ZZ_BARE", protected_attribute="race", correlation=None, risk_level=None)]
    )
    text = drawn_text(svg)

    assert "ZZ_BARE" in text, "the candidate is still listed; it was looked at"
    assert "NOT RATED: 1" in text and "NOT RATED" in text
    assert "LOW: 1" not in text and "LOW" not in text.replace("NOT RATED", ""), (
        "an unrated candidate rendered the emerald LOW badge and was counted as LOW"
    )
    assert "r=0.00" not in text, "a correlation nobody computed rendered as exactly zero"
    assert "not rated for proxy risk" in text


def test_an_unrated_candidate_stays_out_of_the_assessed_denominator():
    """Rule (d). explain._fr_proxy_risk reads len(features) as "among N features"."""
    svg = proxy_risk_to_svg(
        _RATED_PROXIES + [_pv("ZZ_BARE", protected_attribute="race", risk_level=None)]
    )
    finding = str(explanation_meta(svg)["finding"])
    assert "among 2 features" in finding, (
        f"the unrated candidate was counted as assessed and cleared: {finding!r}"
    )


def test_a_rated_candidate_with_no_correlation_keeps_its_level_and_loses_the_number():
    """The level was reported; the magnitude was not. Only the second is withdrawn."""
    svg = proxy_risk_to_svg(
        [
            _pv(
                "zip_code",
                protected_attribute="race",
                correlation=None,
                risk_level=SimpleNamespace(value="high"),
                confidence_score=None,
            )
        ]
    )
    text = drawn_text(svg)
    assert "HIGH: 1" in text and "NOT CHECKED" not in text
    assert "r=0.00" not in text and "r not computed" in text
    assert "not reported" in text, "a 0.5 default drew a half-full confidence bar"
    # The withdrawn bars leave only their empty tracks.
    assert 'opacity="0.8"' not in svg and 'opacity="0.6"' not in svg


def test_a_scan_that_rated_nothing_is_could_not_check():
    svg = proxy_risk_to_svg([_pv("a", risk_level=None), _pv("b", risk_level=None)])
    text = drawn_text(svg)
    assert "NOT CHECKED" in text
    for token in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "CORRELATION"):
        assert token not in text
    assert str(explanation_meta(svg)["finding"]).upper().startswith("COULD NOT CHECK")


def test_a_candidate_carrying_only_its_name_does_not_raise():
    """Missing data is not a programming error; it must reach the NOT RATED row."""
    svg = proxy_risk_to_svg(_RATED_PROXIES + [_pv("ZZ_BARE")])
    assert "NOT RATED: 1" in drawn_text(svg)


def test_every_summary_badge_is_drawn_where_it_can_be_seen():
    """The four pills used to be stacked at x=36; only the last one was visible."""
    svg = proxy_risk_to_svg(
        [
            _pv(
                "zip",
                protected_attribute="g",
                correlation=0.95,
                risk_level="critical",
                confidence_score=0.9,
            ),
            _pv(
                "income",
                protected_attribute="g",
                correlation=0.75,
                risk_level="high",
                confidence_score=0.9,
            ),
            _pv(
                "age_band",
                protected_attribute="g",
                correlation=0.45,
                risk_level="medium",
                confidence_score=0.9,
            ),
            _pv(
                "tenure",
                protected_attribute="g",
                correlation=0.10,
                risk_level="low",
                confidence_score=0.9,
            ),
            _pv("ZZ_BARE", protected_attribute="g", risk_level=None),
        ]
    )
    boxes = badge_boxes(svg)
    labels = [label for _x, label in boxes]
    assert labels == ["CRITICAL: 1", "HIGH: 1", "MEDIUM: 1", "LOW: 1", "NOT RATED: 1"]
    xs = [x for x, _label in boxes]
    assert xs == sorted(xs) and len(set(xs)) == len(xs), (
        f"summary pills overlap, so the severe ones are hidden under the mild ones: {boxes}"
    )
    assert max(xs) < 680, "a pill was pushed off the canvas"


# ── CONTROLS ────────────────────────────────────────────────────────────────
#
# As load-bearing as the guards above. A fix that withdrew a MEASURED row's
# number would be the mirror-image fabrication: silence about a finding that
# was really taken.


def test_control_fully_measured_transformation_keeps_every_number():
    text = drawn_text(
        transformation_comparison_to_svg(
            {"zip_code": 0.72, "income": 0.45}, {"zip_code": 0.28, "income": 0.15}
        )
    )
    for token in ("AVG REDUCTION", "FEATURES IMPROVED", "2 / 2", "Effective", "61%", "66%"):
        assert token in text, f"healthy input lost {token!r}"
    assert "NOT CHECKED" not in text and "NOT MEASURED" not in text


def test_control_fully_populated_intersectional_keeps_its_headline():
    text = drawn_text(
        intersectional_analysis_to_svg(
            {
                "x_attr": "Gender",
                "y_attr": "Race",
                "matrix": {
                    "White": {"Male": 0.65, "Female": 0.72},
                    "Black": {"Male": 0.42, "Female": 0.38},
                },
                "counts": {
                    "White": {"Male": 320, "Female": 310},
                    "Black": {"Male": 180, "Female": 195},
                },
            },
            "loan_approval_rate",
        )
    )
    for token in ("MAX DISPARITY", "0.340", "0.65", "0.38", "n=320"):
        assert token in text, f"healthy input lost {token!r}"
    assert "NOT CHECKED" not in text and "NOT MEASURED" not in text


def test_control_fully_rated_proxy_scan_keeps_its_badges_and_bars():
    svg = proxy_risk_to_svg(_RATED_PROXIES)
    text = drawn_text(svg)
    for token in ("HIGH: 1", "MEDIUM: 1", "zip_code", "r=0.72", "r=0.45", "CONFIDENCE"):
        assert token in text, f"healthy input lost {token!r}"
    assert "NOT CHECKED" not in text and "NOT RATED" not in text
    assert not str(explanation_meta(svg)["finding"]).upper().startswith("COULD NOT CHECK")


def test_control_a_measured_zero_correlation_is_still_a_measurement():
    """r = 0.00 REPORTED is a real result and keeps its number and its bar."""
    svg = proxy_risk_to_svg(
        [
            _pv(
                "tenure",
                protected_attribute="race",
                correlation=0.0,
                risk_level=SimpleNamespace(value="low"),
                confidence_score=0.8,
            )
        ]
    )
    text = drawn_text(svg)
    assert "r=0.00" in text and "LOW: 1" in text
    assert "NOT CHECKED" not in text and "NOT RATED" not in text


def test_control_a_measured_zero_after_correlation_is_a_full_reduction():
    """after = 0.0 REPORTED really is a 100% reduction, and must still say so."""
    text = drawn_text(transformation_comparison_to_svg({"zip_code": 0.72}, {"zip_code": 0.0}))
    assert "100%" in text and "1 / 1" in text
    assert "NOT CHECKED" not in text and "NOT MEASURED" not in text
