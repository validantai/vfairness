"""G11 grading wave: rendering adapters, the monitoring tracker and the
in-processing regularizers, each executed on the degenerate input where the
thing it draws or measures does not exist.

Everything here is a PINNED RENDERED ARTIFACT or a PINNED PENALTY PARAMETER,
never a producer-side data dict. A pin that asserts the producer's own field is
green for the whole class of defect this wave is about, where the number is
computed correctly and then dropped, re-derived or crashed over at the boundary
before a person reads it.

WHAT WAS FOUND AND FIXED, measured on this tree on 2026-09-30.

1. A GUARD THAT COERCES WHILE THE ARITHMETIC DOES NOT, in three of the six
   adapters_fairness canvases. ``_is_finite`` accepts a numeric STRING on
   purpose, and says why in its own docstring: these rows arrive from JSON and
   CSV, where "0.5" is a real measurement that was serialised. The arithmetic
   right below the guard then used the RAW value. On a real report whose
   metrics dict had been through ``json.dumps``:

     metrics_bar_chart_to_svg     TypeError: bad operand type for abs(): 'str'
     effect_sizes_to_svg          TypeError: bad operand type for abs(): 'str'
     confidence_intervals_to_svg  TypeError: '<' not supported between
                                  instances of 'int' and 'str'

   while ``radar_chart_to_svg`` and ``group_comparison_to_svg``, which coerce
   through ``_to_finite_float`` first, rendered the identical report correctly.
   ``"0.5" * 450`` is the worse half: it is a 1350-character string, and it only
   failed later, inside ``max()``. All five now render the stringified report
   BYTE-IDENTICALLY to the numeric one.

2. A CRASHED FINDER PUBLISHED SEVERITY "INFO". ``explain._finding_for``
   answered ``(None, "info")`` when a chart's finder raised. "info" is the
   benign end of the vocabulary and ``metadata()["severity"]`` is what a
   pipeline reads INSTEAD of the sentence, while ``caption()`` falls back to the
   curated concept prose, so the artifact read like a described chart with
   nothing to report, under a confident curated recommendation. Measured:
   ``build_explanation("reliability_diagram", {"ece": "0.5", ...})`` gave
   severity ``info``, finding ``None``, and the action "Recalibrate if ECE
   exceeds roughly 0.05" for a comparison that died on ``'0.5' < 0.05``.

3. THE COUNT OF UNCHECKED METRICS WITHOUT THEIR NAMES, in
   ``adapters_monitoring``. The dashboard printed "N metric(s) requested but
   never checked" while the tracker supplies the NAMES, so a one-letter typo in
   a requested metric was indistinguishable on the canvas from a metric the
   window genuinely could not compute.

CONTROLS. Every case has a healthy twin whose REAL number is asserted, because
a guard that refuses everything passes every refusal test and destroys the unit.
"""

from __future__ import annotations

import copy
import json
import re
import warnings

import numpy as np
import pytest

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness import rendering  # noqa: E402
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer  # noqa: E402
from vfairness.rendering.explain import (  # noqa: E402
    ChartExplanation,
    build_explanation,
)

_TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_STAMP = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d")


def visible_text(svg: str) -> list[str]:
    """Only what a sighted reader reads: the contents of <text> nodes.

    Never the whole document. Searching an SVG for a marker matches attribute
    values and font names, and a match in markup is not something a person sees.
    """
    out = []
    for raw in _TEXT_NODE.findall(svg):
        stripped = _TAG.sub("", raw).strip()
        if stripped:
            out.append(stripped)
    return out


def desc(svg: str) -> str:
    """The accessible one-liner, which is the whole artifact for a screen reader."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "the rendered SVG carries no <desc>"
    return match.group(1)


def metadata(svg: str) -> dict:
    """The machine-readable block a report pipeline parses instead of the prose."""
    match = re.search(r"<metadata[^>]*><!\[CDATA\[(.*?)\]\]></metadata>", svg, re.S)
    assert match, "the rendered SVG carries no <metadata> block"
    return json.loads(match.group(1))


def stable(svg: str) -> str:
    """The rendered artifact with only its wall-clock stamp neutralised."""
    return _STAMP.sub("<STAMP>", svg)


def _render(name, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(rendering, name)(*args, **kwargs)


def _report(y_true, y_pred, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true, y_pred, groups).get_report()


# ── fixtures: one real report, measurably unfair, and its stringified twin ───

_ALT = np.array([1, 0] * 30)
# B is never selected: the most extreme disparity this data can express.
UNFAIR = _report(
    np.concatenate([_ALT, _ALT]),
    np.concatenate([_ALT, np.zeros(60, dtype=int)]),
    np.array(["A"] * 60 + ["B"] * 60),
)
# The SAME report after a JSON or CSV round trip: every metric is text.
STRINGIFIED = copy.deepcopy(UNFAIR)
STRINGIFIED["metrics"] = {
    k: (str(v) if v is not None else None) for k, v in STRINGIFIED["metrics"].items()
}

# The five adapters_fairness canvases in this batch.
FAIRNESS_CHARTS = [
    "radar_chart_to_svg",
    "metrics_bar_chart_to_svg",
    "group_comparison_to_svg",
    "effect_sizes_to_svg",
    "confidence_intervals_to_svg",
]


def test_the_fixture_really_carries_an_extreme_measured_disparity():
    """Without this control every assertion below could pass on data with no gap.

    Derived, not quoted: B receives no positive prediction at all, so the
    demographic parity DIFFERENCE is A's own positive rate and the RATIO is 0.
    """
    a_rate = float(UNFAIR["group_stats"]["A"]["positive_rate"])
    b_rate = float(UNFAIR["group_stats"]["B"]["positive_rate"])
    assert b_rate == 0.0, UNFAIR["group_stats"]
    assert a_rate > 0.0, UNFAIR["group_stats"]
    assert float(UNFAIR["metrics"]["demographic_parity_difference"]) == pytest.approx(
        a_rate - b_rate
    )
    assert float(UNFAIR["metrics"]["demographic_parity_ratio"]) == 0.0
    # And the stringified twin is genuinely text, or the pin below is vacuous.
    assert all(isinstance(v, str) for v in STRINGIFIED["metrics"].values() if v is not None), (
        STRINGIFIED["metrics"]
    )


@pytest.mark.parametrize("chart", FAIRNESS_CHARTS)
def test_a_serialised_report_renders_identically_to_the_numeric_one(chart):
    """The guard accepts a numeric string, so the arithmetic must too.

    Three of these five raised a low-level TypeError on this input while the
    other two rendered it correctly, which is the coerce/arithmetic split in its
    plainest form: same report, same numbers, half the gallery gone.

    Byte-identity (modulo the wall-clock stamp) is the assertion rather than
    "does not raise", because "renders something" is exactly what a fabrication
    also does.
    """
    numeric = stable(_render(chart, UNFAIR))
    text = stable(_render(chart, STRINGIFIED))
    assert numeric == text, (
        f"{chart} does not agree with itself about a report that went through "
        f"JSON: numeric desc={desc(numeric)!r} vs stringified desc={desc(text)!r}"
    )


@pytest.mark.parametrize("chart", FAIRNESS_CHARTS)
def test_the_measured_breach_is_still_on_the_canvas(chart):
    """The over-correction control: a coercion must not mute the finding.

    A guard that refuses everything passes every refusal test and destroys the
    unit, so each canvas has to keep saying something unfavourable about a run
    in which one group is never selected. The severity is read from the
    machine-readable layer, not from prose, and "info" is forbidden.
    """
    svg = _render(chart, UNFAIR)
    severity = metadata(svg)["severity"]
    assert severity != "info", (
        f"{chart} published the benign severity for a report in which group B "
        f"receives no positive prediction at all: {desc(svg)!r}"
    )


def test_a_metric_that_is_text_for_nan_is_still_refused():
    """'nan' the STRING is one of absence's six doors and must not become a number.

    ``float('nan')`` succeeds, so a coercion that stops at "did float() work?"
    turns the literal text 'nan' into a plotted point. The bar chart prints
    "n/a" for it and the radar leaves it off the polygon entirely.
    """
    d = copy.deepcopy(UNFAIR)
    d["metrics"] = dict(d["metrics"])
    d["metrics"]["demographic_parity_difference"] = "nan"
    bars = visible_text(_render("metrics_bar_chart_to_svg", d))
    assert "n/a" in bars, bars[:20]
    # ... and never as the string "nan", which the numeric filters used to print.
    assert not any(re.fullmatch(r"nan%?|-?nan", t) for t in bars), bars


# ── 2. a crashed finder may not publish the benign severity ─────────────────

# The metric fields as text is what a JSON or CSV round trip of a report gives
# you, and it is the shape that kills these finders' comparisons.
_STRINGY = {
    "ece": "0.5",
    "mce": "0.5",
    "brier_score": "0.5",
    "max_disparity": "0.5",
    "fairness_score": "0.5",
}
_FINDERS_THAT_DIE_ON_TEXT = ["reliability_diagram", "intersectional_analysis"]


@pytest.mark.parametrize("template", _FINDERS_THAT_DIE_ON_TEXT)
def test_a_finder_that_raised_does_not_read_as_nothing_to_report(template):
    """A crashed finder is not good news, and it used to say "info".

    Measured before the fix, for ``reliability_diagram``: severity ``info``,
    finding ``None``, caption falling back to the curated concept prose, and the
    action "Recalibrate if ECE exceeds roughly 0.05" printed for a comparison
    that died on ``'0.5' < 0.05``.
    """
    ce = build_explanation(template, _STRINGY)
    assert ce.severity != "info", ce.metadata()
    assert ce.finding, "a chart whose summary could not be computed must say so"
    assert "COULD NOT CHECK" in ce.caption().upper(), ce.caption()
    # The curated recommendation assumes a measurement happened.
    assert "Recalibrate" not in ce.action, ce.action
    assert "could not check" in ce.action.lower(), ce.action
    # The severity a pipeline reads agrees with the sentence a person reads.
    assert ce.metadata()["severity"] == ce.severity != "info"


def test_a_finder_that_worked_keeps_its_own_verdict_and_severity():
    """The over-correction control for the same fix.

    A real reliability finding still states its measured ECE and keeps the
    severity its own finder chose, so the net above cannot be satisfied by
    withholding every verdict.
    """
    ce = build_explanation(
        "reliability_diagram",
        {"ece": 0.01, "mce": 0.02, "n_bins": 10, "n_samples": 500, "calibration_status": "GOOD"},
    )
    assert "COULD NOT CHECK" not in ce.caption().upper(), ce.caption()
    assert "0.010" in ce.caption(), ce.caption()
    assert ce.action.startswith("Recalibrate"), ce.action


def test_chart_explanation_caption_and_paragraph_mint_no_verdict_from_nothing():
    """The two reader-facing methods, executed on an empty explanation.

    ``caption`` leads with the finding when there is one and with the concept's
    first sentence otherwise, and ``paragraph`` joins the non-empty parts. With
    every field blank neither may invent a word, and the severity must be
    printed as given rather than defaulted.
    """
    blank = ChartExplanation(
        template="t", title="T", concept="", reading="", action="", finding=None, severity="medium"
    )
    assert blank.paragraph() == ""
    assert blank.caption() == "T:  (severity: MEDIUM)"
    assert blank.metadata()["finding"] is None
    assert blank.metadata()["severity"] == "medium"

    full = ChartExplanation(
        template="t",
        title="T",
        concept="C one. C two.",
        reading="R.",
        action="A.",
        finding="F.",
        severity="high",
    )
    assert full.paragraph() == "C one. C two. R. F. A."
    assert full.caption() == "T: F. (severity: HIGH)"
    # No finding: the lead is the concept's FIRST sentence, and nothing else.
    no_finding = ChartExplanation(
        template="t",
        title="T",
        concept="C one. C two.",
        reading="R.",
        action="A.",
        finding=None,
        severity="low",
    )
    assert no_finding.caption() == "T: C one. (severity: LOW)"


# ── 3. adapters_calibration: absence in a protected attribute ───────────────

_RNG = np.random.default_rng(0)
_N = 400
_PROB = _RNG.random(_N)
_LABEL = (_RNG.random(_N) < _PROB).astype(int)
_ATTR = np.array(["m", "f"] * (_N // 2))

# Absence reaches a grouping through at least six doors. Each one is a real
# column shape, not a hypothetical: the two string spellings are what an
# ordinary CSV read gives you, and pd.NA is what a nullable pandas dtype gives.
# The names are listed here and the VALUES are built inside the test, because
# building pd.NA at import time would make this module need pandas to collect.
_ABSENCE_DOORS = (
    "python None",
    "float nan",
    "blank string",
    "pd.NA",
    "pd.NaT",
    "literal 'None'",
)


def _attr_with_absence(filler):
    """100 'm', 100 'f', then 200 rows whose group label is *filler*."""
    return np.array((["m"] * 100) + (["f"] * 100) + ([filler] * 200), dtype=object)


def _absence_fillers():
    pd = pytest.importorskip("pandas")
    return {
        "python None": None,
        "float nan": float("nan"),
        "blank string": "",
        "pd.NA": pd.NA,
        "pd.NaT": pd.NaT,
        "literal 'None'": "None",
    }


@pytest.mark.parametrize("door", _ABSENCE_DOORS)
def test_a_missing_group_label_never_becomes_a_demographic_group(door):
    """Six doors, one rule, on the artifact a person reads.

    MEASURED before the fix, with 200 rows of "m" beside 200 whose protected
    attribute was ``None``: the desc read "Calibration disparity 0.012 across 2
    groups; worst m vs best None" at severity MEDIUM, a demographic group named
    after the absence of a demographic label and ranked the better calibrated of
    the two. With ``pd.NA`` the chart did not render at all ("boolean value of
    NA is ambiguous"), and with a float NaN or a ``None`` beside a string it died
    inside ``sorted()``: three different outcomes for one ordinary column with a
    missing cell.
    """
    filler = _absence_fillers()[door]
    svg = _render("group_calibration_to_svg", _LABEL, _PROB, _attr_with_absence(filler))
    labels = visible_text(svg)
    joined = " ".join(labels)

    # No minted name, on the canvas or in the accessible layer.
    for token in ("None", "<NA>", "nan", "NaT"):
        assert not re.search(rf"(?<![A-Za-z<])\b{re.escape(token)}\b(?![A-Za-z>])", joined), (
            f"the {door} door minted a group name on the canvas: "
            f"{[x for x in labels if token in x][:3]}"
        )
    assert "<NA>" not in desc(svg) and "best None" not in desc(svg), desc(svg)
    # And the rows are DISCLOSED rather than silently dropped.
    assert "200 row(s) have no group label" in joined, labels[:8]
    # The two real groups are still compared: the guard withdraws the absent
    # level, not the measurement.
    assert "m" in joined and "f" in joined
    assert "disparity" in desc(svg).lower(), desc(svg)


def test_a_protected_attribute_with_no_missing_cell_renders_unchanged():
    """The over-correction control, asserted as BYTE-IDENTITY.

    The absence guard walks every cell of the column, so the proof that it costs
    the healthy path nothing is that the object-dtype and the string-dtype
    renders of the same data are the same bytes, and that neither carries the
    disclosure clause.
    """
    as_str = stable(_render("group_calibration_to_svg", _LABEL, _PROB, _ATTR))
    as_object = stable(
        _render("group_calibration_to_svg", _LABEL, _PROB, np.array(list(_ATTR), dtype=object))
    )
    assert as_str == as_object
    assert "no group label" not in as_str
    # The real measurement is still there and is a real number.
    assert re.search(r"disparity 0\.0\d\d", desc(as_str)), desc(as_str)


def test_a_missing_group_key_never_becomes_a_ranked_group():
    """The same six doors, one surface along: a MAPPING instead of a column.

    MEASURED before the fix on ``{None: {"ece": 0.02}, "b": {"ece": 0.09}}``:
    "ECE disparity 0.070 (4.5x); worst b vs best None" at severity HIGH, under
    the recommendation "Recalibrate the worst-calibrated group toward parity",
    i.e. toward the calibration of the rows whose group was not recorded.
    """
    pd = pytest.importorskip("pandas")
    for key in (None, float("nan"), "", pd.NA, pd.NaT, "None"):
        svg = _render(
            "calibration_disparity_to_svg",
            {
                "group_metrics": {key: {"ece": 0.02}, "b": {"ece": 0.09}},
                "has_significant_disparity": True,
                "overall_ece": 0.05,
            },
        )
        joined = " ".join(visible_text(svg))
        assert "best None" not in desc(svg) and "<NA>" not in desc(svg), (key, desc(svg))
        # One measured group is left, so the comparison is WITHDRAWN rather than
        # made against a group minted from absence.
        assert "NOT ASSESSABLE" in joined, (key, joined[:200])
        assert "1 key(s) carried no group label" in joined, (key, joined[:200])


def test_two_real_groups_still_produce_the_measured_disparity():
    """Control for the same guard: it must not refuse an ordinary mapping."""
    svg = _render(
        "calibration_disparity_to_svg",
        {
            "group_metrics": {"a": {"ece": 0.02}, "b": {"ece": 0.09}},
            "has_significant_disparity": True,
            "overall_ece": 0.05,
        },
    )
    assert "ECE disparity 0.070 (4.5x); worst b vs best a" in desc(svg), desc(svg)
    assert "carried no group label" not in " ".join(visible_text(svg))


# ── 4. a Pareto frontier searched over a set that was silently smaller ──────


def test_a_dropped_configuration_reaches_the_accessible_layer():
    """The canvas said it and the <desc> did not, which is the whole artifact
    for a screen reader and for anything that parses the SVG.

    MEASURED before the fix, three configurations with one non-finite:
    canvas "1 of 3 configuration(s) have non-finite coordinates and are
    excluded", desc "2 of 2 configurations are Pareto-optimal; best trade-off:
    a. (severity: INFO)". TOTAL loss was already refused; this is the partial
    case, which is the commoner one.
    """
    partial = _render(
        "pareto_frontier_to_svg", [0.1, float("nan"), 0.3], [0.3, 0.2, 0.1], labels=["a", "b", "c"]
    )
    d = desc(partial)
    assert "1 configuration(s) have non-finite coordinates" in d, d
    assert "severity: INFO" not in d, d
    # The denominator is the SURVIVING count, never the supplied one: "2 of 3"
    # would read as a frontier searched over all three.
    assert "of 3 configurations are Pareto-optimal" not in d, d
    # The canvas keeps its own clause too.
    assert "1 of 3 configuration(s) have non-finite coordinates" in " ".join(visible_text(partial))


def test_a_frontier_with_every_coordinate_finite_reads_exactly_as_before():
    """Over-correction control for the same fix."""
    healthy = _render(
        "pareto_frontier_to_svg", [0.1, 0.2, 0.3], [0.3, 0.2, 0.1], labels=["a", "b", "c"]
    )
    d = desc(healthy)
    assert d == (
        "Calibration-Fairness Pareto Frontier: 3 of 3 configurations are Pareto-optimal; "
        "best trade-off: b. (severity: INFO)"
    ), d
    assert "non-finite" not in " ".join(visible_text(healthy))


def test_a_calibration_verdict_from_one_prediction_says_how_many():
    """A badge that rests on one observation must not look like one that does not.

    MEASURED, y_true=[1], y_prob=[0.7]: badge "Poorly Calibrated", "ECE = 0.300",
    desc "Model is Poorly Calibrated (ECE 0.300, MCE 0.300)" at severity HIGH,
    with the sample size nowhere on the canvas. Withholding the badge below
    ``MIN_ROWS_PER_GROUP_FOR_CALIBRATION`` was tried and WITHDRAWN because it
    reddened an existing over-correction control
    (``test_a_partly_unusable_input_is_measured_on_what_survived_and_says_so``,
    a six-row fixture that asserts the verdict is not NOT ASSESSABLE). What
    ships is the count, which conflicts with nothing. See the comment in
    ``adapters_calibration`` for what stays open and why.
    """
    from vfairness._not_assessed import MIN_ROWS_PER_GROUP_FOR_CALIBRATION as _MIN

    tiny = _render("reliability_diagram_to_svg", np.array([1]), np.array([0.7]))
    joined = " ".join(visible_text(tiny))
    assert f"From 1 binned prediction(s), fewer than {_MIN}" in joined, joined[:300]
    assert "provisional" in joined

    # Derived from the shared floor, not from the number 10, so raising the
    # constant cannot leave this test asserting a stale boundary.
    at_floor_labels = np.array([1] * _MIN)
    at_floor_probs = np.full(_MIN, 1.0)
    ok = " ".join(
        visible_text(_render("reliability_diagram_to_svg", at_floor_labels, at_floor_probs))
    )
    assert "provisional" not in ok, ok[:300]
    assert "Well Calibrated" in ok, ok[:300]


def test_the_svg_and_matplotlib_twins_agree_about_absence():
    """The two absence predicates are restated, so they must be pinned together.

    ``adapters_calibration._is_missing_level`` cannot import the matplotlib
    chart's copy: ``vfairness.post_processing`` imports matplotlib at
    package-import time and the SVG renderer is meant to need jinja2 and nothing
    else. Two definitions of "this level is the absence of a value" is exactly
    the shape that drifts (the row floor for the same two charts already drifted
    once, 10 rows against 1, which is why MIN_ROWS_PER_GROUP_FOR_CALIBRATION now
    lives in a third place). So they are compared here, level for level, over
    every door and over ordinary values that must NOT be refused.
    """
    pytest.importorskip("matplotlib", reason="the matplotlib twin needs it to import")
    pd = pytest.importorskip("pandas")
    from vfairness.post_processing.calibration.visualization import (
        _is_missing_level as mpl_missing,
    )
    from vfairness.rendering.adapters_calibration import _is_missing_level as svg_missing

    absent = [None, float("nan"), "", " ", "None", "nan", "NaN", "<NA>", "NaT", pd.NA, pd.NaT]
    present = ["m", "f", "None of the above", 0, 1, 0.0, "0", "unknown"]
    for value in absent:
        assert svg_missing(value) is True, value
        assert mpl_missing(value) == svg_missing(value), value
    for value in present:
        assert svg_missing(value) is False, value
        assert mpl_missing(value) == svg_missing(value), value


# ── 5. adapters.py: the dashboard, the gate and the calibration report ──────


def _dashboard(**over):
    d = {
        "group_stats": {
            "a": {"positive_rate": 0.4, "size": 10},
            "b": {"positive_rate": 0.5, "size": 10},
        },
        "metrics": {"demographic_parity_difference": 0.01},
        "assessment": {"fairness_score": 0.9},
    }
    d.update(over)
    return d


def test_a_dashboard_score_that_is_text_still_renders():
    """The same coerce/arithmetic split, on the panel's headline number.

    MEASURED before the fix, with every field of a report stringified:
    ``raw_score <= 1`` raised ``TypeError: '<=' not supported between instances
    of 'str' and 'int'``, so the whole dashboard was lost for a report that had
    been through json.dumps, while `_is_finite` had already said the value was
    usable.
    """
    numeric = stable(_render("fairness_report_to_svg", _dashboard()))
    text = stable(
        _render(
            "fairness_report_to_svg",
            _dashboard(
                group_stats={
                    "a": {"positive_rate": "0.4", "size": 10},
                    "b": {"positive_rate": "0.5", "size": 10},
                },
                metrics={"demographic_parity_difference": "0.01"},
                assessment={"fairness_score": "0.9"},
            ),
        )
    )
    assert "90/100" in desc(numeric), desc(numeric)
    assert desc(numeric) == desc(text), (desc(numeric), desc(text))


def test_a_group_stats_key_with_no_label_gets_no_bar():
    """A bar drawn for the absence of a group is a stratum that does not exist.

    MEASURED before the fix with ``{None: {"positive_rate": 0.4, "size": 10},
    "b": {...}}``: the panel drew a bar labelled "None" at 40% beside "n=10",
    and with pd.NA one labelled "<NA>". Both the canvas and the accessible layer
    now name the count instead.
    """
    pd = pytest.importorskip("pandas")
    for key in (None, float("nan"), "", pd.NA, pd.NaT, "None"):
        svg = _render(
            "fairness_report_to_svg",
            _dashboard(
                group_stats={
                    key: {"positive_rate": 0.4, "size": 10},
                    "b": {"positive_rate": 0.5, "size": 10},
                }
            ),
        )
        labels = visible_text(svg)
        joined = " ".join(labels)
        assert "1 group key(s) carried no group label" in joined, (key, labels[:10])
        assert "1 group key(s) carried no group label" in desc(svg), (key, desc(svg))
        assert metadata(svg)["severity"] != "info", (key, metadata(svg))
        # No bar row was minted for it: only "b" is plotted.
        assert "n=10" in joined
        assert joined.count("n=10") == 1, labels[:12]


def test_an_ordinary_dashboard_names_no_missing_key():
    """Over-correction control for the same guard."""
    svg = _render("fairness_report_to_svg", _dashboard())
    assert "carried no group label" not in " ".join(visible_text(svg))
    assert "carried no group label" not in desc(svg)
    assert metadata(svg)["severity"] == "info", metadata(svg)


def test_a_withheld_calibration_verdict_is_not_published_as_miscalibrated():
    """A verdict the PRODUCER refused to state, turned into a red breach.

    ``CalibrationReport.is_well_calibrated`` is ``Optional[bool]`` and the
    analyzer sets None deliberately, with its own comment: "`nan < 0.05` is
    False, so an ECE that could not be computed reported 'not well calibrated':
    fail-closed, but still a verdict nobody measured". Its text summary renders
    the same None as "Not assessable (not measured)". The SVG ran it through
    ``bool()``, where None is False, and the template's two-way branch painted
    MISCALIBRATED. MEASURED on a real report with is_well_calibrated=None and a
    finite ECE of 0.047: canvas "MISCALIBRATED", desc "ECE 0.047
    (miscalibrated)" at severity MEDIUM. A fabricated breach costs the same as a
    fabricated all-clear.

    THE REACHABLE DOOR is a report that went through ``to_dict()`` and back,
    which is what a task consumer and a stored artifact do: the analyzer only
    emits None when the ECE is non-finite, i.e. when the chart is NOT ASSESSABLE
    on other grounds, so in-process the two coincide and the branch is masked.
    """
    import copy

    np = pytest.importorskip("numpy")
    from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
    from vfairness.rendering.adapters import calibration_report_to_svg

    rng = np.random.default_rng(0)
    n = 600
    prob = rng.random(n)
    label = (rng.random(n) < prob).astype(int)
    attr = np.array(["m", "f"] * (n // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = CalibrationAnalyzer(label, prob, attr).full_analysis()

    def mutated(**over):
        report = copy.deepcopy(base)
        for key, value in over.items():
            object.__setattr__(report, key, value)
        return report

    def render(report):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return calibration_report_to_svg(report)

    # The control FIRST, so a guard that refuses everything cannot pass this.
    ok = render(base)
    assert ">WELL CALIBRATED<" in ok, visible_text(ok)[:12]
    assert "well calibrated" in desc(ok)
    bad = render(mutated(is_well_calibrated=False))
    assert ">MISCALIBRATED<" in bad, visible_text(bad)[:12]
    assert "miscalibrated" in desc(bad)

    withheld = render(mutated(is_well_calibrated=None))
    assert ">MISCALIBRATED<" not in withheld, visible_text(withheld)[:12]
    assert ">NOT ASSESSABLE<" in withheld, visible_text(withheld)[:12]
    assert "miscalibrated" not in desc(withheld), desc(withheld)
    assert "no calibration verdict was reported" in desc(withheld), desc(withheld)
    # The measured ECE is still reported: only the verdict was withheld.
    assert "ECE 0.047" in desc(withheld), desc(withheld)


def test_a_calibration_group_key_with_no_label_gets_no_row():
    """The same six doors on the per-group calibration table."""
    import copy

    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
    from vfairness.rendering.adapters import calibration_report_to_svg

    rng = np.random.default_rng(0)
    n = 600
    prob = rng.random(n)
    label = (rng.random(n) < prob).astype(int)
    attr = np.array(["m", "f"] * (n // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = CalibrationAnalyzer(label, prob, attr).full_analysis()

    one = list(base.group_metrics.values())[0]
    for key in (None, float("nan"), "", pd.NA, pd.NaT, "None"):
        report = copy.deepcopy(base)
        object.__setattr__(report, "group_metrics", {key: one, **base.group_metrics})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = calibration_report_to_svg(report)
        joined = " ".join(visible_text(svg))
        for token in ("None", "<NA>", "nan", "NaT"):
            assert not re.search(rf"(?<![A-Za-z<])\b{re.escape(token)}\b(?![A-Za-z>])", joined), (
                key,
                [x for x in visible_text(svg) if token in x][:3],
            )
        assert "1 group key(s) carried no group label" in desc(svg), (key, desc(svg))


def test_the_gate_relays_the_producers_status_and_shows_the_missing_number():
    """A CI gate must not RE-DERIVE pass/fail, and must not hide a missing value.

    The adapter relays ``status`` and carries the metric value through the
    engine's last door, so a NaN metric prints "N/A" rather than the string
    "nan", and a bool prints "N/A" rather than "1.0000" (the best value on a
    difference axis). Executed across nan, a numeric string, True, None and an
    ordinary float.
    """
    from enum import Enum
    from types import SimpleNamespace

    from vfairness.rendering.adapters import cicd_pipeline_to_svg

    class _S(Enum):
        passed = "passed"

    class _G(Enum):
        approved = "approved"

    gate = SimpleNamespace(status=_G.approved, approved=True, blocking_reasons=[], warnings=[])
    val = SimpleNamespace(passed=True, errors=[], warnings=[])

    def row(value):
        return SimpleNamespace(
            test_name="t1", status=_S.passed, metric_name="DP", metric_value=value, threshold=0.10
        )

    def shown(value):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svg = cicd_pipeline_to_svg(
                validation_result=val, test_results=[row(value)], gate_decision=gate
            )
        return visible_text(svg)

    assert "0.080" in shown(0.08)
    assert "0.080" in shown("0.08"), "a serialised number is a real measurement"
    for absent in (float("nan"), True):
        labels = shown(absent)
        assert "N/A" in labels, (absent, labels[:14])
        assert not any(re.fullmatch(r"nan|1\.0000|1\.000", x) for x in labels), (absent, labels)


# ── 6. rendering.engine: the last door before the markup ────────────────────


def test_every_template_named_by_list_templates_really_exists():
    """list_templates and get_template_path, executed against each other.

    `list_templates` globs the directory, so the only way it can lie is by
    naming something `get_template_path` cannot open. The count is asserted as
    "more than a handful" rather than pinned to today's 46, because a quoted
    number goes stale and then makes deleting a template the cheapest way to
    green.
    """
    from vfairness.rendering.engine import get_template_path, list_templates

    names = list_templates()
    assert len(names) > 20, names
    assert names == sorted(names), "the listing is documented as sorted"
    for name in names:
        assert get_template_path(name).exists(), name
    # And a name it does NOT list is refused, with the available set in the
    # message rather than a bare KeyError.
    with pytest.raises(FileNotFoundError) as err:
        get_template_path("no_such_template")
    assert "Available:" in str(err.value)


@pytest.mark.parametrize(
    "bad", ["../engine", "a b", "a/b", "a.b", "a-b", "", "..%2fengine", "templates/../engine"]
)
def test_render_svg_refuses_a_template_name_that_is_a_path(bad):
    """The traversal guard, asserted as a refusal rather than as a rendered file."""
    from vfairness.rendering.engine import render_svg

    with pytest.raises(ValueError, match="Invalid template_name"):
        render_svg(bad, {})


def test_a_bar_length_is_never_drawn_for_a_value_that_is_not_a_number():
    """A LENGTH IS A MEASUREMENT A READER COMPARES BY EYE.

    ``bar_width`` was the one numeric filter in the engine that did not go
    through ``_to_float``, although that function's docstring calls itself the
    last door for every one of them. MEASURED 2026-09-30:
    ``bar_width(True)`` was 300, the FULL bar and therefore the highest rate on
    the panel, while the label beside it printed "N/A"; ``bar_width(nan)`` was 4,
    identical to a measured rate of 0.0, because ``nan > 4`` is False so
    ``max()`` returned the floor; and ``bar_width("0.5")`` and
    ``bar_width(None)`` both raised TypeError, taking the whole chart down for a
    rate whose LABEL renders perfectly well.
    """
    pd = pytest.importorskip("pandas")
    from vfairness.rendering.engine import _bar_width

    # Measured values, including a real zero, keep their length.
    assert _bar_width(0.5, 300) == 150.0
    assert _bar_width(1.0, 300) == 300
    assert _bar_width(2.0, 300) == 300, "clamped, not extrapolated"
    assert _bar_width(0.0, 300) == 4, "a measured zero keeps the visible stub"
    assert _bar_width("0.5", 300) == 150.0, "a serialised number is a measurement"
    # Absence gets NO bar, which is a third state distinct from the 4px stub.
    for absent in (None, float("nan"), float("inf"), True, False, pd.NA, pd.NaT, "", "nan", []):
        assert _bar_width(absent, 300) == 0, absent


def test_a_label_is_never_minted_out_of_an_absent_value():
    """``str(x)`` on an absent value MINTS content, in ~90 label slots.

    MEASURED: ``truncate_text(None)`` returned the string "None" and
    ``truncate_text(float('nan'))`` returned "nan", so a group, a feature or a
    finding whose name was absent got one, in the font and the position of a
    real label.
    """
    pd = pytest.importorskip("pandas")
    from vfairness.rendering.engine import _truncate_text

    for absent in (None, float("nan"), pd.NA, pd.NaT):
        assert _truncate_text(absent) == "N/A", absent
    # A STRING is passed through unchanged, including the literal "None": this
    # filter also renders free prose, and a caller who supplied text supplied
    # text. Group LEVELS are screened at their own boundary instead.
    assert _truncate_text("None") == "None"
    assert _truncate_text("") == ""
    assert _truncate_text("a name") == "a name"
    assert _truncate_text("x" * 60, 50) == "x" * 50 + "…"
    assert _truncate_text(5) == "5"


def test_an_absent_severity_is_not_drawn_as_the_clean_tick():
    """``str(None).lower()`` is "none", which is a live KEY of the icon map.

    MEASURED: ``_severity_icon(None)`` returned "✓", the mark this library draws
    for a clean row. It is the identical trap ``adapters._module_score`` carries
    a comment about, one module along, and in the louder direction.
    """
    pd = pytest.importorskip("pandas")
    from vfairness.rendering.engine import _severity_icon

    for absent in (None, float("nan"), pd.NA, pd.NaT):
        assert _severity_icon(absent) == "?", absent
    # A severity the detector actually REPORTED keeps its mark.
    assert _severity_icon("none") == "✓"
    assert _severity_icon("critical") == "✗"
    assert _severity_icon("low") == "○"
    assert _severity_icon("medium") == "⚠"


def test_a_flag_is_never_banded_as_a_risk_score():
    """``True >= 0.75`` is True, so a bool painted the red HIGH band.

    The mirror of the green-for-nothing case the slate branch exists for, and
    the same class as ``f4(True)`` printing "1.0000". ``_is_nan`` let a bool
    through because a bool is not NaN.
    """
    pd = pytest.importorskip("pandas")
    from vfairness.rendering.engine import _is_nan, _risk_bg, _risk_color, _risk_label

    # Controls first: a real score keeps its real band.
    assert _risk_label(0.9) == "HIGH"
    assert _risk_label(0.0) == "MINIMAL"
    assert _risk_label("0.5") == "MEDIUM", "a serialised number is a measurement"
    assert _risk_color(0.9) == "#dc2626"
    for absent in (None, float("nan"), True, False, pd.NA, pd.NaT, "", "nan"):
        assert _risk_label(absent) == "N/A", absent
        assert _risk_color(absent) == "#64748b", absent
        assert _risk_bg(absent) == "#f1f5f9", absent
    # And `_is_nan` now RETURNS A BOOL. It used to return pd.NA itself, because
    # `False or pd.NA` yields the right operand as it is, so the caller's `if`
    # raised "boolean value of NA is ambiguous" from a line unrelated to it and
    # the except here caught nothing.
    for absent in (None, float("nan"), pd.NA, pd.NaT):
        assert _is_nan(absent) is True, absent
    for present in (0.0, 0.5, "x"):
        assert _is_nan(present) is False, present


# ── 7. adapters_experimentation: an unlabelled subgroup row ─────────────────

_EXP_RESULT = dict(
    overall_effect=0.12,
    overall_ci=(0.05, 0.19),
    overall_p_value=0.004,
    intersection_effects={
        ("a",): {"effect": 0.1, "p_value": 0.02, "n": 300},
        ("b",): {"effect": 0.2, "p_value": 0.03, "n": 300},
    },
    heterogeneity_detected=True,
    heterogeneity_p_value=0.01,
    power_results=[{"intersection": ("a",), "power": 0.9, "is_powered": True}],
    n_intersections=2,
    n_excluded=0,
    design_type="factorial",
    metadata={},
)


@pytest.mark.parametrize("door", _ABSENCE_DOORS)
def test_a_subgroup_row_with_no_label_is_not_given_one(door):
    """`.get(key, default)` DOES NOT FIRE when the key is PRESENT holding None.

    MEASURED 2026-09-30: ``power_analysis_to_svg([{"intersection": None, ...}])``
    skipped the ``("unknown",)`` default and ``str(None)`` minted the label
    "None", so the canvas drew a row and a bar labelled "None" with a real power
    percentage beside it, in the font and the position of a real subgroup.
    pd.NA and pd.NaT print "<NA>" and "NaT" through the same line.
    """
    filler = _absence_fillers()[door]
    svg = _render(
        "power_analysis_to_svg", [{"intersection": filler, "power": 0.9, "is_powered": True}]
    )
    labels = visible_text(svg)
    joined = " ".join(labels)
    for token in ("None", "<NA>", "nan", "NaT"):
        assert not re.search(rf"(?<![A-Za-z<])\b{re.escape(token)}\b(?![A-Za-z>])", joined), (
            door,
            [x for x in labels if token in x][:3],
        )
    # It gets the SAME word an absent key already got, so the row is still
    # visible and still says it has no name.
    assert "unknown" in joined, labels[:12]
    # The measured power is untouched.
    assert "90%" in joined, labels[:12]


def test_a_named_subgroup_keeps_its_name():
    """Over-correction control: the tuple label is still joined as before."""
    svg = _render(
        "power_analysis_to_svg", [{"intersection": ("a", "b"), "power": 0.9, "is_powered": True}]
    )
    assert "a × b" in " ".join(visible_text(svg))


def test_an_effect_row_with_no_label_is_not_given_one():
    """The same `.get` trap in the sibling results table."""
    svg = _render(
        "experiment_results_to_svg",
        {
            **_EXP_RESULT,
            "intersection_effects": [
                {"intersection": None, "effect": 0.1, "p_value": 0.02, "n": 300}
            ],
        },
    )
    labels = visible_text(svg)
    assert not any(re.fullmatch(r"None", x) for x in labels), labels[:14]
    assert "unknown" in " ".join(labels), labels[:14]


def test_the_experimentation_renders_refuse_an_empty_call():
    """All four, executed with nothing to draw. None may claim a result."""
    for name, args in (
        ("power_analysis_to_svg", (None,)),
        ("experiment_results_to_svg", (None,)),
        ("experiment_recommendation_to_svg", ({},)),
        ("causal_decomposition_to_svg", ()),
    ):
        svg = _render(name, *args)
        assert "COULD NOT CHECK" in desc(svg).upper(), (name, desc(svg))
        assert metadata(svg)["severity"] != "info", (name, metadata(svg))


def test_the_experimentation_renders_state_a_real_verdict_when_given_one():
    """The controls, so none of the refusals above can be satisfied by refusing
    everything."""
    powered = desc(
        _render(
            "power_analysis_to_svg",
            [
                {"intersection": ("a",), "power": 0.9, "is_powered": True},
                {"intersection": ("b",), "power": 0.42, "is_powered": False},
            ],
        )
    )
    assert "1 of 2 intersections are underpowered" in powered, powered

    results = desc(_render("experiment_results_to_svg", _EXP_RESULT))
    assert "+0.1200" in results and "p=0.0040" in results, results

    rec = desc(
        _render(
            "experiment_recommendation_to_svg",
            {"decision": "DEPLOY_TREATMENT", "confidence": 0.9, "reasoning": ["x"]},
        )
    )
    assert "Deploy Treatment (confidence 90%)" in rec, rec

    causal = desc(
        _render(
            "causal_decomposition_to_svg",
            {"total_effect": 0.2, "direct_effect": 0.1, "indirect_effect": 0.1},
        )
    )
    assert "COULD NOT CHECK" in causal.upper(), (
        "a decomposition with no proportion mediated must say so: " + causal
    )


def test_a_withheld_experimentation_number_is_never_printed_as_a_value():
    """Each of the four, on a NaN where its headline number goes."""
    nan = float("nan")
    # One row, no power and no verdict: the whole chart refuses, and says that
    # the refusal is NOT a finding that every intersection is powered.
    power = desc(_render("power_analysis_to_svg", [{"intersection": ("a",), "power": nan}]))
    assert "COULD NOT CHECK" in power.upper(), power
    assert "not a finding that every intersection is adequately powered" in power, power
    assert not re.search(r"avg power \d", power), power
    # A row WITH a verdict beside a row whose power is NaN: the average is
    # withheld rather than computed over a zero that nobody measured.
    partial = desc(
        _render(
            "power_analysis_to_svg",
            [{"intersection": ("a",), "power": nan, "is_powered": True}],
        )
    )
    assert "avg power not reported" in partial, partial
    # ... and not as a percentage. "target 80%" legitimately contains "0%", so
    # the assertion is on the SLOT, not on the two characters.
    assert not re.search(r"avg power \d", partial), partial

    results = desc(_render("experiment_results_to_svg", {**_EXP_RESULT, "overall_p_value": nan}))
    assert "p=not computed" in results, results

    rec = desc(
        _render(
            "experiment_recommendation_to_svg",
            {"decision": "DEPLOY_TREATMENT", "confidence": nan},
        )
    )
    assert "confidence not reported" in rec, rec
    assert not re.search(r"confidence \d", rec), rec


# ── 8. operations.monitoring: the window record and the monitor's config ────


def _window(metrics, alerts, **over):
    from datetime import datetime

    from vfairness.operations.monitoring.tracker import WindowMetrics

    kwargs = dict(
        batch_id="b",
        timestamp=datetime(2026, 9, 30, 10, 0),
        sample_count=600,
        metrics=metrics,
        group_rates={},
        alerts=alerts,
    )
    kwargs.update(over)
    return WindowMetrics(**kwargs)


def test_an_alert_entry_that_is_present_holding_none_is_not_a_comparison():
    """`k not in alerts` is a MEMBERSHIP question and this is a VALUE question.

    MEASURED 2026-09-30 with ``metrics={"a": 0.1}, alerts={"a": None}``:

        any_alert            False   (a clean verdict)
        n_compared           1
        uncompared_metrics   []      ("everything was compared")

    a full clean bill of health over a guardrail that was never applied, while
    the two SVG renderers drawn from this SAME object read the identical window
    as NOT MONITORED, because both of them test
    ``name in alerts and alerts[name] is not None``. Two surfaces disagreeing
    about one window, and the one a JSON reader meets first answered "clean".

    ``FairnessMonitor`` omits the key rather than writing None, so the reachable
    door is a window that was BUILT rather than computed: a stored window read
    back, a consumer's own record, a caller assembling one by hand.
    """
    withheld = _window({"a": 0.1}, {"a": None})
    assert withheld.any_alert is None, "nothing was compared, so there is no verdict"
    assert withheld.uncompared_metrics == ["a"]
    assert withheld.to_dict()["n_compared"] == 0
    assert withheld.to_dict()["any_alert"] is None

    # The MIXED window is the one that proves the guard is not a blanket
    # refusal: one metric compared and clean, one never compared.
    mixed = _window({"a": 0.1, "b": 0.2}, {"a": None, "b": False})
    assert mixed.any_alert is False, "the comparison that ran came back clean"
    assert mixed.uncompared_metrics == ["a"]
    assert mixed.to_dict()["n_compared"] == 1


def test_a_fully_compared_window_still_reports_its_real_verdict():
    """The over-correction control for the guard above, in all three states."""
    clean = _window({"a": 0.1}, {"a": False})
    assert clean.any_alert is False
    assert clean.uncompared_metrics == []
    assert clean.to_dict()["n_compared"] == 1

    breached = _window({"a": 0.9}, {"a": True})
    assert breached.any_alert is True
    assert breached.uncompared_metrics == []

    nothing = _window({}, {})
    assert nothing.any_alert is None, "an empty record is could-not-check, never clean"

    # A metric computed but never compared (the tracker's own shape for a NaN
    # metric: the key is OMITTED from alerts) is unchanged by the fix.
    absent = _window({"a": float("nan")}, {})
    assert absent.any_alert is None
    assert absent.uncompared_metrics == ["a"]


def test_the_window_json_boundary_carries_the_scope_with_the_verdict():
    """to_dict is the boundary a consumer reads INSTEAD of the object."""
    d = _window({"a": 0.1, "b": float("nan")}, {"a": False}).to_dict()
    assert d["any_alert"] is False
    assert d["n_compared"] == 1
    assert d["uncompared_metrics"] == ["b"], d
    # Every field of the dataclass reaches the dict: a field-by-field rebuild
    # that lists the old fields and not the new ones is how a disclosure added
    # upstream is silently dropped.
    import dataclasses

    from vfairness.operations.monitoring.tracker import WindowMetrics

    for f in dataclasses.fields(WindowMetrics):
        assert f.name in d, f"to_dict drops the field {f.name!r}"


def test_a_monitor_config_says_so_when_it_cannot_monitor_anything():
    """A typo and a vacuous bound both used to be accepted in total silence.

    MEASURED 2026-09-30: ``FairnessMonitorConfig`` validated nothing, so
    ``metrics_to_track=["demografic_parity"]`` (one letter wrong) computed no
    metric, compared nothing to a threshold, and produced zero warnings at
    ingest; and ``alert_threshold=-1.0`` was accepted although no value either
    comparison can see could breach it, so the monitor reports a clean window
    forever.
    """
    from vfairness.operations.monitoring.tracker import (
        MONITORABLE_METRICS,
        FairnessMonitorConfig,
    )

    def warnings_for(**kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FairnessMonitorConfig(**kw)
            return [str(w.message) for w in caught]

    # THE CONTROL FIRST. The library default and the four-fifths rule must be
    # silent, or this guard is one that argues for breaking the product.
    assert warnings_for() == []
    assert warnings_for(alert_threshold=0.8) == []
    assert warnings_for(alert_threshold=1.0) == []
    assert warnings_for(metrics_to_track=["equalized_odds"]) == []
    assert warnings_for(metrics_to_track=sorted(MONITORABLE_METRICS)) == []

    typo = warnings_for(metrics_to_track=["demografic_parity"])
    assert len(typo) == 1, typo
    assert "demografic_parity" in typo[0] and "cannot compute" in typo[0]
    # The supported set is NAMED, so the reader can see what they meant.
    for known in MONITORABLE_METRICS:
        assert known in typo[0], (known, typo[0])

    empty = warnings_for(metrics_to_track=[])
    assert len(empty) == 1 and "is empty" in empty[0], empty

    # A bound no result can breach, on BOTH of the two comparisons this one
    # number drives. The reason text comes from the shared
    # `_metric_direction.vacuous_bound_reason`, so it names the range.
    vacuous = warnings_for(alert_threshold=0.0)
    assert len(vacuous) == 2, vacuous
    assert any("disparate_impact" in w and "no result can fall below it" in w for w in vacuous)
    assert any("demographic_parity" in w and "no result can exceed it" in w for w in vacuous)

    # The MIRROR: a band no result can satisfy, so every window alerts.
    always = warnings_for(alert_threshold=5.0)
    assert len(always) == 1 and "True for every window" in always[0], always

    assert any("not a usable count" in w for w in warnings_for(window_size=0))
    assert any("not a usable count" in w for w in warnings_for(min_samples=0))
    assert any("negative" in w for w in warnings_for(alert_cooldown_seconds=-1))


def test_the_monitorable_set_matches_what_update_and_check_branches_on():
    """The guard on the guard: a fifth metric added to the dispatch and not to
    the set would make every mention of it a false "cannot compute" warning."""
    from pathlib import Path

    from vfairness.operations.monitoring import tracker as T
    from vfairness.operations.monitoring.tracker import MONITORABLE_METRICS

    source = Path(T.__file__).read_text(encoding="utf-8")
    branched = set(re.findall(r'"(\w+)" in self\.config\.metrics_to_track', source))
    branched |= set(
        re.findall(r'for m in \("(\w+)", "(\w+)"\)', source)[0]
        if re.findall(r'for m in \("(\w+)", "(\w+)"\)', source)
        else []
    )
    assert branched, "the dispatch no longer tests metrics_to_track by name"
    assert branched <= set(MONITORABLE_METRICS), sorted(branched - set(MONITORABLE_METRICS))


def test_a_drift_result_reports_a_shift_only_when_both_means_exist():
    """DriftResult and MultiscaleDriftResult, executed on the degenerate cases."""
    from datetime import datetime

    from vfairness.operations.monitoring.drift import DriftResult, MultiscaleDriftResult

    def result(**over):
        kwargs = dict(
            metric="m",
            scale="short",
            ks_statistic=0.3,
            p_value=0.01,
            drift_score=0.5,
            drift_detected=True,
            reference_mean=0.4,
            current_mean=0.5,
            reference_n=100,
            current_n=100,
        )
        kwargs.update(over)
        return DriftResult(**kwargs)

    # Control: the real shift, derived rather than quoted.
    healthy = result()
    assert healthy.mean_shift == pytest.approx(healthy.current_mean - healthy.reference_mean)
    assert healthy.to_dict()["comparison_ran"] is True

    # A mean that does not exist gives a shift that does not exist, and the
    # dict carries a NaN rather than a 0.0 that reads as "no movement".
    for over in ({"reference_mean": float("nan")}, {"current_mean": float("nan")}):
        r = result(**over)
        assert math_isnan(r.mean_shift), over
        assert math_isnan(r.to_dict()["mean_shift"]), over

    # worst_scale ranks by drift score and puts NaN scores LAST, so a scale
    # that could not be scored never outranks one that was.
    def multi(scales, **over):
        kwargs = dict(
            metric="m",
            timestamp=datetime(2026, 9, 30),
            scales=scales,
            overall_drift_score=0.5,
            drift_detected=True,
        )
        kwargs.update(over)
        return MultiscaleDriftResult(**kwargs)

    assert multi({}).worst_scale is None, "no scale is not a scale with no drift"
    ranked = multi({"s": result(drift_score=0.2), "l": result(drift_score=0.8)})
    assert ranked.worst_scale.drift_score == 0.8
    # The NaN must lose whichever order it arrives in: `nan > x` and `x > nan`
    # are both False, so in max() a NaN wins or loses by argument order alone.
    for scales in (
        {"nan_first": result(drift_score=float("nan")), "real": result(drift_score=0.3)},
        {"real": result(drift_score=0.3), "nan_last": result(drift_score=float("nan"))},
    ):
        assert multi(scales).worst_scale.drift_score == 0.3, scales


def math_isnan(value):
    import math

    return isinstance(value, float) and math.isnan(value)


def test_an_alert_payload_carries_its_own_unmeasured_fields_as_they_are():
    """AlertPayload: a record, not a measurement, and it must not round one."""
    from datetime import datetime

    from vfairness.operations.monitoring.alerts import AlertPayload

    def payload(**over):
        kwargs = dict(
            alert_id="a1",
            timestamp=datetime(2026, 9, 30),
            severity="high",
            priority_score=0.9,
            metric_name="dp",
            affected_groups=["a"],
            drift_score=0.5,
            mean_shift=0.2,
            routing={"team": "x"},
            drift_event={},
            message="m",
        )
        kwargs.update(over)
        return AlertPayload(**kwargs)

    p = payload()
    assert p.acknowledged is False and p.resolution is None
    p.acknowledge("fixed")
    assert p.acknowledged is True and p.resolution == "fixed"
    # acknowledge() with no note still flips the flag and leaves the note absent
    # rather than inventing one.
    q = payload()
    q.acknowledge()
    assert q.acknowledged is True and q.resolution is None

    # An unmeasurable score stays unmeasurable through the serialisation: a 0.0
    # here would read as "no drift and the lowest priority".
    nan = payload(priority_score=float("nan"), drift_score=float("nan"), mean_shift=float("nan"))
    d = nan.to_dict()
    for key in ("priority_score", "drift_score", "mean_shift"):
        assert math_isnan(d[key]), (key, d[key])
    # Every field of the dataclass reaches the dict, EXCEPT the one whose
    # omission is ADJUDICATED with a reason in tests/test_serialiser_honesty.py.
    # That registry is the repo's disposition of this exact question and it
    # already says why: drift_event is this payload's INPUT, and the reading
    # derived from it (metric_name, drift_score, mean_shift, affected_groups,
    # severity, priority_score) is emitted in full. So the assertion is on the
    # registry, not against it: any OTHER field that stops reaching the dict is
    # the silent-drop defect, and this one may only stay out while the registry
    # still carries its reason.
    import dataclasses

    from tests.test_serialiser_honesty import ADJUDICATED

    registered = set(
        ADJUDICATED.get("vfairness.operations.monitoring.alerts.AlertPayload.to_dict", {})
    )
    assert registered == {"drift_event"}, registered
    emitted = payload().to_dict()
    for f in dataclasses.fields(AlertPayload):
        if f.name in registered:
            continue
        assert f.name in emitted, f.name


def test_the_uncompared_metrics_are_named_and_not_only_counted():
    """The observation this batch was handed, closed.

    ``adapters_monitoring.alert_timeline_to_svg`` printed the COUNT of metrics a
    window never compared ("2 metric(s) never checked") while
    ``WindowMetrics.uncompared_metrics`` supplies the NAMES, so a REQUESTED
    metric whose name has a one-letter typo was indistinguishable on the canvas
    from a metric the window genuinely could not compute. The typo is precisely
    the case that produces no other signal anywhere: nothing validated
    ``metrics_to_track`` (that half is fixed in FairnessMonitorConfig, above).

    The fixture uses the British spelling of a real metric, which is what an
    operator actually types.
    """
    from vfairness.rendering.adapters_monitoring import alert_timeline_to_svg

    typo = "equalised_odds_group_gender"  # the library computes "equalized_"
    window = _window(
        {"demographic_parity_group_gender": 0.02, typo: float("nan")},
        {"demographic_parity_group_gender": False},
    )
    assert window.uncompared_metrics == [typo], window.uncompared_metrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = alert_timeline_to_svg([window])

    # The NAME is in the timeline ROW CELL, asserted as its own text node
    # rather than as a substring of the whole canvas: the explanation panel
    # repeats the <desc>, so a whole-canvas search would pass on the desc alone
    # and say nothing about the cell. (Verified: sabotaging only the cell left a
    # whole-canvas assertion green.)
    cells = visible_text(svg)
    assert any(t == typo or t.startswith(typo + " +") for t in cells), cells[:14]
    # ... and in the accessible layer, which is the whole artifact for a screen
    # reader and for anything that parses the SVG.
    assert typo in desc(svg), desc(svg)
    assert "severity: INFO" not in desc(svg), desc(svg)

    # The fully compared control keeps its clean sentence and its INFO severity,
    # so the disclosure is a function of the data and not printed unconditionally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = alert_timeline_to_svg(
            [
                _window(
                    {"demographic_parity_group_gender": 0.02},
                    {"demographic_parity_group_gender": False},
                )
            ]
        )
    assert "Never compared" not in desc(clean), desc(clean)
    assert "(1 clean)" in desc(clean) and "severity: INFO" in desc(clean), desc(clean)


# ── 9. in_processing.regularizers: the MITIGATION's own parameter ───────────
#
# A regularizer's failure mode is INVERTED. A penalty that has stopped working
# produces a BETTER fairness number, not a worse one, because the model is left
# as it was (or pushed the other way) while the dependence measure keeps
# measuring correctly and keeps saying measured=True. So these pins read the
# GRADIENT of the penalty with respect to the predictions, never the fairness
# number the regularizer reports.

torch = pytest.importorskip("torch", reason="the regularizers need the [torch] extra")


def _dependent_batch(n=200):
    """Predictions that depend on the sensitive attribute BY CONSTRUCTION.

    Rates 0.2 and 0.8, so every regularizer here has a real dependence to
    penalise. Without that, a penalty of 0.0 would be the correct answer and
    every assertion below would pass on a batch with nothing to fix.
    """
    torch.manual_seed(0)
    sensitive = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    pred = (0.2 + 0.6 * sensitive + 0.05 * torch.randn(n)).clamp(0.01, 0.99)
    label = (torch.rand(n) < 0.5).float()
    return pred, sensitive, label


REGULARIZERS = [
    "StatisticalParityRegularizer",
    "GroupFairnessRegularizer",
    "CorrelationPenalty",
    "HilbertSchmidtRegularizer",
]


def _penalty_gradient(class_name, strength):
    """|d(penalty)/d(y_pred)| for one regularizer at one strength."""
    from vfairness.in_processing.regularizers import fairness_regularizers as R

    pred, sensitive, label = _dependent_batch()
    pred = pred.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = getattr(R, class_name)(strength=strength)
        out = reg(pred, sensitive, label)
    penalty = out[0] if isinstance(out, tuple) else out
    penalty.backward()
    grad = 0.0 if pred.grad is None else float(pred.grad.abs().sum())
    return float(penalty.detach()), grad


@pytest.mark.parametrize("class_name", REGULARIZERS)
def test_a_working_penalty_has_a_nonzero_gradient_toward_less_dependence(class_name):
    """The control, and the only thing that makes the two below non-vacuous."""
    penalty, grad = _penalty_gradient(class_name, 0.1)
    assert penalty > 0.0, (class_name, penalty)
    assert grad > 0.0, (class_name, grad)


@pytest.mark.parametrize("class_name", REGULARIZERS)
def test_a_strength_of_zero_is_a_mitigation_that_applies_nothing_and_says_so(class_name):
    """MEASURED: strength 0.0 gives penalty 0.000000 and |grad| exactly 0.0.

    The loss is unchanged, the optimiser is told nothing, and
    ``RegularizerMetrics`` still reports a real dependence with
    ``measured=True``, so a run configured this way is indistinguishable
    downstream from a regularised one. It was accepted in total silence.
    """
    from vfairness.in_processing.regularizers import fairness_regularizers as R

    penalty, grad = _penalty_gradient(class_name, 0.0)
    assert penalty == 0.0 and grad == 0.0, (class_name, penalty, grad)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(R, class_name)(strength=0.0)
        messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert "APPLIES NO MITIGATION" in messages[0], messages[0]
    assert class_name in messages[0]


@pytest.mark.parametrize("class_name", REGULARIZERS)
def test_a_negative_strength_is_an_anti_fairness_term_and_says_so(class_name):
    """The worse half. MEASURED at strength -0.5: penalty -0.147234 with
    |grad| 0.500000 pointing the OTHER WAY, so minimising the total loss
    MAXIMISES the dependence between the predictions and the sensitive
    attribute, under a fairness name."""
    from vfairness.in_processing.regularizers import fairness_regularizers as R

    penalty, grad = _penalty_gradient(class_name, -0.5)
    assert penalty < 0.0, (class_name, penalty)
    assert grad > 0.0, "the gradient is not absent, it is reversed"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(R, class_name)(strength=-0.5)
        messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert "REVERSES" in messages[0] and "anti-fairness" in messages[0], messages[0]


def test_a_strength_that_is_not_a_usable_number_says_so_at_construction():
    """The remaining degenerate parameters, and the types that look fine."""
    from decimal import Decimal

    import numpy as np

    from vfairness.in_processing.regularizers import fairness_regularizers as R

    def messages(strength):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            R.StatisticalParityRegularizer(strength=strength)
            return [str(w.message) for w in caught]

    # CONTROLS FIRST. A real strength, an int, and the numpy scalars a training
    # script routinely produces must all be silent, or this guard is one that
    # argues for breaking the product.
    for fine in (0.1, 1.0, 1, np.float32(0.2), np.float64(0.3)):
        assert messages(fine) == [], fine

    assert any("non-finite" in m for m in messages(float("nan")))
    assert any("non-finite" in m for m in messages(float("inf")))
    assert any("a bool is not a regularization strength" in m for m in messages(True))
    assert any("a bool is not a regularization strength" in m for m in messages(False))
    # A numeric STRING is the trap: float("0.1") succeeds, so a convertibility
    # test passes it, and `tensor * "0.1"` still raises inside forward().
    assert any("not a real number" in m for m in messages("0.1")), messages("0.1")
    assert any("not a real number" in m for m in messages(Decimal("0.1")))
    assert any("not a real number" in m for m in messages(None))


def test_base_regularizer_is_abstract_and_hands_out_a_copy_of_its_history():
    """BaseRegularizer itself and get_history, executed."""
    from vfairness.in_processing.regularizers.fairness_regularizers import BaseRegularizer

    base = BaseRegularizer(strength=0.1)
    assert base.get_history() == []
    pred, sensitive, _ = _dependent_batch(4)
    with pytest.raises(NotImplementedError):
        base.forward(pred, sensitive)

    # A COPY, so a caller mutating the returned list cannot rewrite the record
    # a later reader grades the run by.
    handed_out = base.get_history()
    handed_out.append("not a RegularizerMetrics")
    assert base.get_history() == []

    # And a real history is handed out in order, by value.
    from vfairness.in_processing.regularizers.fairness_regularizers import (
        StatisticalParityRegularizer,
    )

    reg = StatisticalParityRegularizer(strength=0.1)
    pred, sensitive, label = _dependent_batch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            reg(pred, sensitive, label, return_metrics=True)
    history = reg.get_history()
    assert len(history) == 3, history
    assert all(h.measured for h in history)
    reg.reset_history()
    assert reg.get_history() == []
    assert len(history) == 3, "the previously handed-out list is the caller's own"


def test_regularizer_metrics_carries_its_third_state_through_the_dict():
    """RegularizerMetrics and to_dict: a record, and it must not lose `measured`.

    0.0 is the CLEAN end of every dependence scale in this module (HSIC is zero
    iff independent, a Pearson r of 0.0 is no information at all), so a
    `measured` flag that does not survive serialisation turns a refusal into the
    best possible finding.
    """
    import dataclasses
    import math

    from vfairness.in_processing.regularizers.fairness_regularizers import RegularizerMetrics

    measured = RegularizerMetrics(regularization_value=0.05, dependence_measure=0.3)
    d = measured.to_dict()
    assert d["measured"] is True
    assert d["dependence_measure"] == 0.3
    assert d["regularization_value"] == 0.05

    refused = RegularizerMetrics(
        regularization_value=0.0,
        dependence_measure=float("nan"),
        measured=False,
        metadata={"not_assessed": "a single-valued sensitive attribute"},
    )
    r = refused.to_dict()
    assert r["measured"] is False
    assert math.isnan(r["dependence_measure"]), "a refusal may not become a clean 0.0"
    # The penalty value stays a real 0.0: it describes the tensor that WAS added
    # to the loss, which is a true statement about the optimisation step.
    assert r["regularization_value"] == 0.0
    assert r["metadata"]["not_assessed"]
    for f in dataclasses.fields(RegularizerMetrics):
        assert f.name in r, f"to_dict drops the field {f.name!r}"


# ── 10. in_processing.calibrators: the trainer and its state record ─────────


def _tiny_trainer(epochs=1):
    import torch.nn as nn

    from vfairness.in_processing.calibrators.group_calibrators import (
        CalibrationAwareTrainer,
        TrainableGroupCalibrator,
    )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 1)

        def forward(self, x):
            return self.layer(x).squeeze(-1)

    torch.manual_seed(0)
    model = _Model()
    calibrator = TrainableGroupCalibrator(n_groups=2)
    return (
        CalibrationAwareTrainer(model, calibrator, calibration_epochs=epochs),
        model,
        calibrator,
    )


def _batch(n=64):
    torch.manual_seed(1)
    x = torch.randn(n, 4)
    y = (torch.rand(n) < 0.5).float()
    groups = torch.cat(
        [torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)]
    )
    return x, y, groups


def test_a_calibration_term_that_was_not_applied_is_nan_and_never_zero():
    """0.0 on this scale is a PERFECTLY CALIBRATED batch.

    Three different things used to report it: include_calibration=False, a
    calibrator that refused because no group reached its minimum size, and group
    ids outside range. Executed on all three, plus the healthy control.
    """
    import math

    import torch.nn as nn

    x, y, groups = _batch()

    def step(**kw):
        trainer, model, calibrator = _tiny_trainer()
        opt = torch.optim.SGD(list(model.parameters()) + list(calibrator.parameters()), lr=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return trainer.train_step(
                kw.pop("x", x),
                kw.pop("y", y),
                kw.pop("groups", groups),
                opt,
                nn.BCEWithLogitsLoss(),
                **kw,
            )

    # CONTROL: a real batch produces a real, finite calibration loss that is
    # actually added to the objective.
    healthy = step()
    assert not math.isnan(healthy["calibration_loss"])
    assert healthy["calibration_loss"] > 0.0
    assert healthy["total_loss"] == pytest.approx(
        healthy["task_loss"] + healthy["calibration_loss"], abs=1e-6
    )

    off = step(include_calibration=False)
    assert math.isnan(off["calibration_loss"]), off
    assert off["total_loss"] == pytest.approx(off["task_loss"], abs=1e-9)

    out_of_range = step(groups=torch.full((x.shape[0],), 5, dtype=torch.long))
    assert math.isnan(out_of_range["calibration_loss"]), out_of_range

    tiny = step(x=x[:2], y=y[:2], groups=groups[:2])
    assert math.isnan(tiny["calibration_loss"]), tiny
    # A refused term is LEFT OUT of the objective rather than poisoning every
    # gradient with NaN.
    assert not math.isnan(tiny["total_loss"])


def test_fine_tuning_that_measured_nothing_returns_nan_and_warns():
    """The class docstring binds this return to the name ``ece``, so a 0.0 here
    was read as a measured calibration error nobody computed."""
    import math

    x, y, groups = _batch()
    loader = [(x[:32], y[:32], groups[:32]), (x[32:], y[32:], groups[32:])]

    def run(epochs, batches):
        trainer, _model, calibrator = _tiny_trainer(epochs)
        opt = torch.optim.SGD(calibrator.parameters(), lr=0.01)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = trainer.fine_tune_calibration(batches, opt)
            return value, [str(w.message) for w in caught]

    # CONTROL: a real loader returns a real mean over the measured batches.
    value, _ = run(1, loader)
    assert not math.isnan(value) and value > 0.0, value

    for label, epochs, batches in (
        ("epochs=0", 0, loader),
        ("epochs=-1", -1, loader),
        ("empty loader", 1, []),
        ("every batch refused", 1, [(x[:2], y[:2], groups[:2])]),
    ):
        value, messages = run(epochs, batches)
        assert math.isnan(value), (label, value)
        assert any("nothing was measured" in m for m in messages), (label, messages)
        assert any("0.0 would read as perfect calibration" in m for m in messages), label


def test_calibration_state_defaults_its_global_error_to_nan_not_zero():
    """CalibrationState: a record. Its one numeric default is the third state."""
    import dataclasses
    import math

    from vfairness.in_processing.calibrators.group_calibrators import CalibrationState

    bare = CalibrationState(method="temperature", parameters={"T": 1.2})
    assert math.isnan(bare.global_ece), "a default of 0.0 is a perfect calibration"
    assert bare.group_ece == {} and bare.group_ece_post == {}
    assert math.isnan(bare.to_dict()["global_ece"])

    full = CalibrationState(
        method="temperature",
        parameters={"T": 1.2},
        group_ece={"a": 0.02},
        group_ece_post={"a": 0.01},
        global_ece=0.03,
    )
    d = full.to_dict()
    assert d["global_ece"] == 0.03
    assert d["group_ece"] == {"a": 0.02} and d["group_ece_post"] == {"a": 0.01}
    for f in dataclasses.fields(CalibrationState):
        assert f.name in d, f"to_dict drops the field {f.name!r}"


# ── 11. xai.storage: a bulk write that landed in part ──────────────────────


def _stub_writer(data):
    """A SupabaseWriter around a stub client.

    ``__init__`` is bypassed on purpose, exactly as tests/test_bgl3_xai_1.py
    does: it imports ``supabase``, an optional extra that is not installed here,
    and the methods under test only ever touch ``self._client``. The stub is
    imported from that file rather than restated, so the two cannot disagree
    about what a PostgREST response looks like.
    """
    from tests.test_bgl3_xai_1 import _writer

    return _writer(data)


def _explanation(i):
    from vfairness.xai.schemas import Attribution, Explanation

    return Explanation(
        method="shap",
        instance_id=f"row-{i}",
        subject_id="s",
        model_hash="m",
        data_hash="d",
        base_value=0.5,
        prediction=0.6,
        attributions=[Attribution(feature="a", contribution=0.1)],
        units="logit",
        scope="local",
    )


def test_a_bulk_write_that_landed_in_part_is_refused_like_one_that_landed_at_all():
    """PARTIAL loss passed where TOTAL loss was refused.

    ``if not resp.data`` checks the count at zero and nowhere else. MEASURED
    2026-09-30 against the stub client, ten Explanation rows in:

        10 returned -> 10 ids, no warning       (correct)
         0 returned -> SupabaseWriterError naming "0 of 10"
         3 returned ->  3 ids, NO WARNING

    A worker calling write_explanations got a shorter list back and nothing told
    it that seven explanations are not in the database. The length is not a
    signal the caller can read either: it never knew how many rows the producer
    had, and the natural next step, storing the ids, succeeds with three.
    """
    from vfairness.xai.storage.supabase_writer import SupabaseWriterError

    ten = [_explanation(i) for i in range(10)]

    # CONTROL FIRST: a complete write returns every id and warns about nothing.
    complete = _stub_writer([{"id": f"id-{k}"} for k in range(10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids = complete.write_explanations("owner", ten)
    assert len(ids) == 10 and ids[0] == "id-0"
    assert [str(w.message) for w in caught] == []

    # An empty input writes nothing and says so by returning nothing: that is
    # the correct answer, not a refusal.
    assert _stub_writer([]).write_explanations("owner", []) == []

    with pytest.raises(SupabaseWriterError, match="returned no data"):
        _stub_writer([]).write_explanations("owner", ten)

    with pytest.raises(SupabaseWriterError) as partial:
        _stub_writer([{"id": f"id-{k}"} for k in range(3)]).write_explanations("owner", ten)
    assert "only 3 of 10" in str(partial.value), str(partial.value)
    assert "NOT in the database" in str(partial.value)
    # The refusal must not echo the attempted row data, which is the reason the
    # neighbouring refusals withhold the response repr.
    assert "row-0" not in str(partial.value)

    # MORE rows back than were sent loses nothing, so it warns rather than
    # raising: the response does not describe the request.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids = _stub_writer([{"id": f"id-{k}"} for k in range(12)]).write_explanations("owner", ten)
    assert len(ids) == 12
    assert any("returned 12 row(s) for 10 sent" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_a_written_row_with_no_primary_key_is_named_not_a_bare_keyerror():
    """`row["id"]` raised KeyError: 'id' from a line naming neither the table
    nor the position, so a caller could not tell it from a bug of its own."""
    from vfairness.xai.schemas import FairnessDecomposition
    from vfairness.xai.storage.supabase_writer import SupabaseWriterError

    with pytest.raises(SupabaseWriterError) as many:
        _stub_writer([{"id": "a"}, {"nope": "b"}, {"id": "c"}]).write_explanations(
            "owner", [_explanation(i) for i in range(3)]
        )
    assert "position 1 of 3" in str(many.value) and "no 'id' key" in str(many.value)

    decomposition = FairnessDecomposition(
        id="x",
        subject_id="s",
        metric="demographic_parity",
        protected_attribute="g",
        total_disparity=0.3,
        per_feature={"a": 0.2, "b": 0.1},
        proxy_scores={"a": 0.5},
        flagged_proxies=["a"],
        audit_artifact_id="aa",
    )
    with pytest.raises(SupabaseWriterError, match="no 'id' key"):
        _stub_writer([{"nope": "x"}]).write_fairness_decomposition("owner", decomposition)

    # CONTROL: the same single-row write succeeds and returns the id.
    assert _stub_writer([{"id": "d1"}]).write_fairness_decomposition("owner", decomposition) == "d1"


def test_a_decomposition_whose_identity_is_broken_never_reaches_the_database():
    """write_fairness_decomposition goes through to_db_row, which re-asserts the
    1e-6 identity, so a row whose per-feature shares do not sum to the total is
    refused BEFORE the insert rather than stored and read back later."""
    from vfairness.xai.schemas import FairnessDecomposition

    broken = FairnessDecomposition(
        id="x",
        subject_id="s",
        metric="demographic_parity",
        protected_attribute="g",
        total_disparity=0.9,  # per_feature sums to 0.3
        per_feature={"a": 0.2, "b": 0.1},
        proxy_scores={"a": 0.5},
        flagged_proxies=["a"],
        audit_artifact_id="aa",
    )
    writer = _stub_writer([{"id": "d1"}])
    with pytest.raises(ValueError, match="identity broken"):
        writer.write_fairness_decomposition("owner", broken)
    # Nothing was sent: the client log carries no insert.
    assert not any(entry[0] == "insert" for entry in writer._client.log), writer._client.log


def test_supabase_writer_error_is_a_runtime_error_and_the_writer_needs_its_env():
    """SupabaseWriterError and the constructor, executed."""
    import os

    from vfairness.xai.storage.supabase_writer import SupabaseWriter, SupabaseWriterError

    assert issubclass(SupabaseWriterError, RuntimeError)
    try:
        raise SupabaseWriterError("boom")
    except RuntimeError as exc:
        assert str(exc) == "boom"

    # The constructor refuses BEFORE building a client, and it is the missing
    # optional extra that is reported here rather than a bare ImportError from
    # deep inside supabase-py. This is why every test above bypasses __init__.
    saved = {k: os.environ.pop(k, None) for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")}
    try:
        with pytest.raises((ImportError, SupabaseWriterError)) as err:
            SupabaseWriter()
        assert "supabase" in str(err.value) or "SUPABASE_URL" in str(err.value)
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


# ── 12. legal.admissibility ─────────────────────────────────────────────────


def test_the_legal_mappers_answer_absence_the_way_they_answer_none():
    """Four of absence's six doors CRASHED inside `_normalise`.

    MEASURED 2026-09-30: ``map_domain_to_use_case(float('nan'))`` raised
    ``AttributeError: 'float' object has no attribute 'lower'``,
    ``map_jurisdiction(pd.NA)`` raised ``TypeError: boolean value of NA is
    ambiguous`` from the ``if not`` itself, and an int or a bool raised the same
    AttributeError. A missing cell in a CSV or a nullable pandas column is
    exactly how a domain and a jurisdiction arrive.

    The answer is the one None already got, not a new policy.
    """
    pd = pytest.importorskip("pandas")
    from vfairness.legal.admissibility import map_domain_to_use_case, map_jurisdiction

    # CONTROLS FIRST: the real mappings must be untouched.
    assert map_domain_to_use_case("hiring") == "recruitment"
    assert map_domain_to_use_case("HIRING") == "recruitment"
    assert map_domain_to_use_case(" hiring ") == "recruitment"
    assert map_domain_to_use_case("lending") == "lending"
    assert map_jurisdiction("us") == "us-federal"
    assert map_jurisdiction("united states") == "us-federal"
    assert map_jurisdiction("EU") == "eu"
    assert map_jurisdiction("germany") == "de"
    # An unknown but REAL string keeps its documented fallback.
    assert map_domain_to_use_case("nonsense") == "generic"
    assert map_jurisdiction("nonsense") is None

    for absent in (None, "", "   ", float("nan"), pd.NA, pd.NaT, 123, True):
        assert map_domain_to_use_case(absent) == "generic", absent
        assert map_jurisdiction(absent) is None, absent


def test_classify_columns_refuses_a_single_string_instead_of_its_characters():
    """A bare string IS a Sequence[str] and never the intent.

    MEASURED: ``classify_columns("race", "recruitment", "us")`` iterated the
    CHARACTERS and produced four findings over the columns 'r', 'a', 'c' and
    'e', each counted into ``summary.unknown``, so a caller who passed one
    column name instead of a list got a legal block about four columns that do
    not exist.
    """
    from vfairness.legal.admissibility import classify_columns

    with pytest.raises(TypeError, match="takes a SEQUENCE"):
        classify_columns("race", "recruitment", "us")

    # CONTROL: the list form is what it was.
    block = classify_columns(["race"], "recruitment", "us")
    assert len(block["findings"]) == 1
    assert block["findings"][0]["column"] == "race"


def test_classify_columns_never_names_a_column_after_the_absence_of_a_name():
    """`_normalise(None)` raised AttributeError inside the matcher, so ONE
    missing header took the whole legal block down; `str(None)` would have put
    a column called "None" in a block a lawyer reads."""
    pd = pytest.importorskip("pandas")
    from vfairness.legal.admissibility import classify_columns

    for absent in (None, float("nan"), "", "   ", pd.NA, pd.NaT):
        block = classify_columns([absent, "race"], "recruitment", "us")
        assert block["unmappedColumns"] == ["(unnamed column)"], (absent, block)
        named = [f["column"] for f in block["findings"]]
        assert named == ["race"], (absent, named)
        for token in ("None", "nan", "<NA>", "NaT"):
            assert token not in json.dumps(block), (absent, token)


def test_classify_columns_determines_a_real_status_and_withholds_an_unknown_one():
    """The whole point of the unit, and the control for the two guards above.

    A rule pack that answers nothing would pass every refusal test here, so the
    pin asserts a REAL breakdown, derived from the pack rather than quoted: the
    same five attributes must produce a determined status under a jurisdiction
    the pack covers, and an UNKNOWN one must be counted as unknown and never as
    allowed.
    """
    from vfairness.legal.admissibility import classify_columns

    columns = ["race", "sex", "age", "disability", "religion"]
    covered = classify_columns(columns, "recruitment", "us")
    assert covered["coverage"] == "covered", covered["coverage"]
    determined = (
        covered["summary"]["forbidden"]
        + covered["summary"]["restricted"]
        + covered["summary"]["monitoringOnly"]
        + covered["summary"]["allowed"]
    )
    assert determined > 0, covered["summary"]
    assert covered["summary"]["forbidden"] >= 1, covered["summary"]
    assert covered["jurisdictionId"] == "us-federal"
    assert covered["sourceRevision"]

    # With no jurisdiction there is no pack, so every mapped attribute is
    # UNKNOWN. It may never be counted as allowed: that is the whole rule.
    nowhere = classify_columns(columns, "recruitment", None)
    assert nowhere["coverage"] == "uncovered"
    assert nowhere["summary"]["allowed"] == 0, nowhere["summary"]
    assert nowhere["summary"]["unknown"] == len(columns), nowhere["summary"]
    assert all(f["status"] == "unknown" for f in nowhere["findings"])


# ── 13. llm.text_fairness ───────────────────────────────────────────────────


def test_the_text_fairness_records_carry_their_coverage_and_their_nans():
    """GroupScore, GroupScore.to_dict and TextFairnessResult, executed.

    These are records rather than measurements, so what is checked is that they
    mint nothing and drop nothing: a group with no usable text keeps a NaN mean
    rather than a 0.0, and the two coverage fields (the only machine-readable
    trace of the identity terms nothing was scored for) survive the hand-written
    key list in TextFairnessResult.to_dict.
    """
    import dataclasses
    import math

    from vfairness.llm.text_fairness import GroupScore, TextFairnessResult

    scored = GroupScore(group="a", n=30, mean_score=0.8, gap_vs_overall=0.05)
    assert scored.to_dict() == {
        "group": "a",
        "n": 30,
        "mean_score": 0.8,
        "gap_vs_overall": 0.05,
    }

    empty = GroupScore(group="a", n=0, mean_score=float("nan"), gap_vs_overall=float("nan"))
    d = empty.to_dict()
    assert d["n"] == 0
    assert math.isnan(d["mean_score"]) and math.isnan(d["gap_vs_overall"]), d
    for f in dataclasses.fields(GroupScore):
        assert f.name in d, f.name

    result = TextFairnessResult(
        overall_mean=0.8,
        groups=[GroupScore("a", 30, 0.85, 0.05), GroupScore("b", 30, 0.75, -0.05)],
        worst_group="b",
        max_gap=0.10,
        p_value=0.03,
        severity="medium",
        interpretation="x",
        groups_not_scored=["c", "d"],
        n_groups_supplied=4,
    )
    r = result.to_dict()
    # The hand-written key list must carry EVERY field: it does not fail when
    # one is added, it just stops carrying it, and these two are the caveat.
    for f in dataclasses.fields(TextFairnessResult):
        assert f.name in r, f"to_dict drops the field {f.name!r}"
    assert r["groups_not_scored"] == ["c", "d"]
    assert r["n_groups_supplied"] == 4
    assert len(r["groups"]) == 2 and r["groups"][0]["group"] == "a"
    # The lists are COPIES, so a consumer mutating them cannot rewrite the
    # record a later reader grades the run by.
    r["groups_not_scored"].append("e")
    r["notes"].append("invented")
    assert result.groups_not_scored == ["c", "d"]
    assert result.notes == []

    # A run that scored nothing keeps a NaN gap and a None worst group rather
    # than a 0.0 gap, which on this scale is perfect agreement.
    nothing = TextFairnessResult(
        overall_mean=float("nan"),
        groups=[],
        worst_group=None,
        max_gap=float("nan"),
        p_value=None,
        severity="not_assessed",
        interpretation="y",
    )
    n = nothing.to_dict()
    assert math.isnan(n["max_gap"]) and math.isnan(n["overall_mean"])
    assert n["worst_group"] is None and n["p_value"] is None
    assert n["groups"] == []
