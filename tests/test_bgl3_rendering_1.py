"""A chart may not certify parity it reached by dropping a group.

BGL3 rendering batch. Six rendering units were graded by execution against
inputs where the quantity they draw genuinely does not exist. Five refuse
honestly and are pinned here so they cannot stop. One did not, and it is the
shape the brief calls the fourth bullet: a group silently dropped from a
comparison, so the comparison reports agreement among whatever remained.

THE DEFECT, MEASURED. A 130-row classification report: A (60 rows) and B (60
rows) selected identically, so every disparity really is 0.0 between them, and
C (10 rows) never selected at all, the most extreme unfairness the data can
express. C falls below the default ``min_group_size=30``, so the engine drops it
before any metric runs and says so: ``report_assessability`` resolved
``{'assessable': True, 'reason': '', 'excluded': ['C (n=10)']}`` and every
adapter passed that list on to its template as ``excluded_groups``. No template
reads that key (grep the templates: zero hits), and ``_excluded_clause`` in
explain.py was reachable only from ``_could_not_check``, i.e. only on a run that
measured NOTHING. So on this run, which measured two thirds of the data, the
shipped artifacts read:

    radar_chart_to_svg          canvas "✓ Fair"
                                desc   "Overall fairness status: Fair." INFO
    metrics_bar_chart_to_svg    canvas "PASSED 5 / 5", "FAIR RATE 100%"
                                desc   "5 of 5 checked fairness metric(s)
                                        passed." INFO
    disparity_heatmap_to_svg    canvas "✓ Low disparity across groups"
                                desc   "Low disparity across groups across 2
                                        group(s) and 5 metric(s)." INFO
    effect_sizes_to_svg         canvas "|d|=0.00 (Negligible)"
                                desc   "Largest effect |d|=0.00
                                        (Negligible)." INFO
    group_comparison_to_svg     canvas named C (its own clause)
                                desc   "Max disparity ..." with no mention of C
    confidence_intervals_to_svg canvas no mention of C
                                desc   named C only because this run had no
                                        interval to report at all

A perfect all-clear, on the two surfaces a reader actually gets, produced by
removing the only group that could have made it unequal. The fairness score for
that run is 1.0.

BOTH HALVES ARE ASSERTED. Every case below has a HEALTHY TWIN: the same two
large groups with no third group at all. A renderer that printed an exclusion
notice unconditionally would pass the first half of this file and fail the
second, so the disclosure has to be a function of the data.

WHAT IS READ. The contents of ``<text>`` nodes for the canvas, and the
``<desc>`` plus the ``<metadata>`` JSON for the accessible layer. Never the
whole document: searching an SVG for a marker matches attribute values and font
names (``n/a`` matches "Verdana"), and a match in markup is not something a
person sees.
"""

from __future__ import annotations

import dataclasses
import json
import re
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer
from vfairness.rendering.adapters_fairness import (
    COULD_NOT_CHECK,
    FAIL,
    NOT_ASSESSABLE_TEXT,
    PASS,
    _state_style,
    report_assessability,
)
from vfairness.rendering.engine import JINJA2_MISSING_MESSAGE, raise_jinja2_missing
from vfairness.rendering.explain import (
    ChartExplanation,
    build_explanation,
    inject_accessibility,
)
from vfairness.rendering.skins import apply_skin

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness import rendering  # noqa: E402

# The six adapters in adapters_fairness.py that resolve a report's assessability
# and therefore receive the dropped-group list.
CHARTS = [
    "radar_chart_to_svg",
    "metrics_bar_chart_to_svg",
    "disparity_heatmap_to_svg",
    "effect_sizes_to_svg",
    "group_comparison_to_svg",
    "confidence_intervals_to_svg",
]

TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


def visible_text(svg: str) -> list[str]:
    """Only what a sighted reader reads: the contents of <text> nodes."""
    out = []
    for raw in TEXT_NODE.findall(svg):
        stripped = TAG.sub("", raw).strip()
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


def _report(y_true, y_pred, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true, y_pred, groups).get_report()


def _render(name, report_dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(rendering, name)(report_dict)


# A and B are selected identically, so the disparity between them is a real 0.0.
# C is never selected AND is below the default min_group_size of 30, so it is
# dropped before any metric runs. The sizes are written against that default
# rather than as bare numbers being "small", so the fixture cannot silently
# become measurable.
_N_LARGE = 60
_N_DROPPED = 10
assert _N_DROPPED < 30, "C must fall below the default min_group_size to be dropped"
_ALTERNATING = np.array([1, 0] * (_N_LARGE // 2))

DROPPED = _report(
    np.concatenate([_ALTERNATING, _ALTERNATING, np.array([1, 0] * (_N_DROPPED // 2))]),
    np.concatenate([_ALTERNATING, _ALTERNATING, np.zeros(_N_DROPPED, dtype=int)]),
    np.array(["A"] * _N_LARGE + ["B"] * _N_LARGE + ["C"] * _N_DROPPED),
)
# The healthy twin: the same two groups, measured the same way, with no third
# group in the data at all. Nothing was dropped, so nothing may be disclosed.
HEALTHY = _report(
    np.concatenate([_ALTERNATING, _ALTERNATING]),
    np.concatenate([_ALTERNATING, _ALTERNATING]),
    np.array(["A"] * _N_LARGE + ["B"] * _N_LARGE),
)

_DROPPED_NAME = "C (n=%d)" % _N_DROPPED


# ── the defect ──────────────────────────────────────────────────────────────


def test_the_fixture_really_does_hide_an_extreme_disparity():
    """Without this the rest of the file could pass on data with no gap to hide.

    Measured: fairness_score 1.0, demographic_parity_difference 0.0 and
    demographic_parity_ratio 1.0 over A and B, while C was never selected once.
    """
    assess = report_assessability(DROPPED)
    assert assess["assessable"] is True, "the run over A and B is genuinely measurable"
    assert assess["excluded"] == [_DROPPED_NAME], assess
    assert DROPPED["assessment"]["fairness_score"] == 1.0, DROPPED["assessment"]
    metrics = DROPPED["metrics"]
    assert metrics["demographic_parity_difference"] == 0.0
    assert float(metrics["demographic_parity_ratio"]) == 1.0
    assert report_assessability(HEALTHY)["excluded"] == [], "the twin drops nothing"


@pytest.mark.parametrize("chart", CHARTS)
def test_a_group_dropped_before_any_comparison_is_named_on_the_canvas(chart):
    """Five of these six canvases named nothing at all. Measured before the fix:
    the radar drew "✓ Fair", the bar chart "PASSED 5 / 5" and "FAIR RATE 100%",
    the heatmap "✓ Low disparity across groups", the effect-size tiles "|d|=0.00
    (Negligible)" and the forest plot its own fixed legend, none of them
    mentioning C. Only group_comparison_to_svg named it, in its own clause.

    The forest plot was the trap in this count: a first probe scored it as
    disclosing because the word "excluded" appears in its subtitle, in a
    sentence about METRICS without a computed interval. The check is on the
    group's own name.
    """
    labels = visible_text(_render(chart, DROPPED))
    joined = " ".join(labels)

    assert _DROPPED_NAME in joined, (
        f"{chart} drew a verdict over two of three groups and never named the "
        f"third, whose 10 rows were never selected once: {labels[:24]}"
    )
    assert "insufficient evidence" in joined, (
        f"{chart} names the group without saying why it is absent, which reads "
        f"as a group with nothing to report: {labels[:24]}"
    )


@pytest.mark.parametrize("chart", CHARTS)
def test_a_group_dropped_before_any_comparison_reaches_the_accessible_layer(chart):
    """The <desc> is the whole artifact for a screen-reader user.

    Measured before the fix, on the same report: "Overall fairness status:
    Fair. (severity: INFO)", "5 of 5 checked fairness metric(s) passed.
    (severity: INFO)", "Low disparity across groups across 2 group(s) and 5
    metric(s). (severity: INFO)" and "Largest effect |d|=0.00 (Negligible);
    average |d|=0.00. (severity: INFO)". The severity is asserted because a
    pipeline reads THAT instead of the sentence, and "info" there is the same
    unqualified all-clear the sentence is forbidden to make.
    """
    svg = _render(chart, DROPPED)

    assert _DROPPED_NAME in desc(svg), (
        f"the desc certifies over a group it never named: {desc(svg)}"
    )
    meta = metadata(svg)
    assert _DROPPED_NAME in str(meta["finding"]), meta["finding"]
    assert meta["severity"] != "info", (
        f"{chart} reports a benign severity for a run that never compared one of "
        f"its groups: {meta['severity']} / {meta['finding']}"
    )


@pytest.mark.parametrize("chart", CHARTS)
def test_a_chart_that_dropped_nothing_says_nothing_about_exclusions(chart):
    """The over-correction control: the disclosure is a function of the data.

    Same two groups, same identical selection rates, no third group. A chart
    that cried "insufficient evidence" here would be read once and ignored
    afterwards, and it would pass every assertion above.
    """
    svg = _render(chart, HEALTHY)
    joined = " ".join(visible_text(svg))

    assert "insufficient evidence" not in joined, (
        f"{chart} claims a group was excluded from a report that excluded none: {joined[:300]}"
    )
    assert "Excluded:" not in desc(svg), desc(svg)
    # The severity half applies only where the chart HAS a verdict to report.
    # confidence_intervals_to_svg is red here for a reason of its own and was
    # red before this batch: get_report() computes no bootstrap interval unless
    # asked (include_ci=True), so that chart genuinely has no significance to
    # claim on either report and refuses on both. Asserting "info" for it would
    # have meant weakening a refusal that is correct.
    finding = str(metadata(svg)["finding"] or "")
    if "COULD NOT CHECK" not in finding.upper():
        assert metadata(svg)["severity"] == "info", (
            f"{chart} withholds an all-clear it measured in full: {metadata(svg)}"
        )


def test_build_explanation_appends_the_clause_once_and_only_once():
    """The not-assessable path already wrote it, via ``_could_not_check``.

    Before the fix the clause appeared ONLY there, so this is the one case that
    was already correct and the one place a second append would double the
    sentence. Asserted by count, not by presence.
    """
    data = {
        "not_assessable": True,
        "not_assessable_reason": "1 comparable group(s) after filtering (2 are needed).",
        "excluded_groups": ["C (n=10)"],
    }

    finding = build_explanation("fairness_report", data).finding

    assert finding.count("Excluded: C (n=10).") == 1, finding


def test_build_explanation_leaves_a_fully_measured_finding_byte_identical():
    """An empty ``excluded_groups`` must change nothing, not even the severity.

    This is how the over-correction control is guaranteed rather than hoped: the
    two explanations are compared field by field.
    """
    measured = {
        "fairness_score": 92,
        "n_passed": 5,
        "n_total": 5,
        "metrics": [{"passed": True}] * 5,
    }

    with_key = build_explanation("fairness_report", dict(measured, excluded_groups=[]))
    without_key = build_explanation("fairness_report", measured)

    assert with_key == without_key
    assert "Excluded" not in str(with_key.finding), with_key.finding
    assert with_key.severity == "info", with_key.severity


# ── the five that already refuse ────────────────────────────────────────────


def test_report_assessability_resolves_three_states_and_not_two():
    """Verified refusal. Executed on real engine reports, not hand-built dicts.

    A 125-row report whose 25-row minority falls below min_group_size=30 leaves
    ONE comparable group, so no between-group comparison ran at all: it returns
    assessable False with a reason naming the excluded group. The three-group
    report above stays assessable (two groups really were compared) and still
    carries the dropped one. A bare dict with no assessment block claims nothing
    either way and is deliberately NOT turned into a failure: inventing one
    would be the mirror-image lie.
    """
    maj, minority = 100, 25
    y_pred = np.concatenate([np.array([1] * 56 + [0] * 44), np.zeros(minority, dtype=int)])
    y_true = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] * 12 + [0] * 13)])
    one_group = _report(y_true, y_pred, np.array(["maj"] * maj + ["min"] * minority))

    refused = report_assessability(one_group)
    assert refused["assessable"] is False
    assert NOT_ASSESSABLE_TEXT in refused["reason"]
    assert "min (n=25)" in refused["reason"], refused["reason"]
    assert refused["excluded"] == ["min (n=25)"]

    measured = report_assessability(DROPPED)
    assert measured["assessable"] is True
    assert measured["excluded"] == [_DROPPED_NAME]

    assert report_assessability({}) == {"assessable": True, "reason": "", "excluded": []}


def test_raise_jinja2_missing_raises_and_never_returns_a_document():
    """Verified refusal. 16 of 44 adapters used to warn once and return "".

    An empty string written to a .svg file is a zero-byte report that reads as
    this run's output, and Python shows a given warning once per process, so a
    batch job produced nothing at all, silently, after the first.
    """
    with pytest.raises(ImportError) as excinfo:
        raise_jinja2_missing()

    assert str(excinfo.value) == JINJA2_MISSING_MESSAGE
    assert "vfairness[rendering]" in JINJA2_MISSING_MESSAGE, (
        "the message must name the extra that installs the backend"
    )


def test_metadata_publishes_every_field_of_the_explanation():
    """Verified refusal, plus the structural half.

    ``metadata()`` is a hand-written field list, which is the shape that drops a
    later field: a disclosure computed correctly in one function and absent from
    the only path a consumer reads. A new field on ChartExplanation with no
    entry here fails this test rather than disappearing.
    """
    published = {
        "template": "chart",
        "title": "title",
        "concept": "concept",
        "reading": "howToRead",
        "action": "recommendation",
        "finding": "finding",
        "severity": "severity",
    }
    fields = {f.name for f in dataclasses.fields(ChartExplanation)}
    assert fields == set(published), (
        f"ChartExplanation gained or lost a field; decide whether a consumer "
        f"needs it and update metadata() and this map: {fields ^ set(published)}"
    )

    refused = build_explanation(
        "fairness_report",
        {"not_assessable": True, "not_assessable_reason": "nothing was measured."},
    )
    meta = refused.metadata()
    for field, key in published.items():
        assert meta[key] == getattr(refused, field), key
    assert COULD_NOT_CHECK.replace("_", " ").upper() in meta["finding"].upper(), meta["finding"]
    assert meta["severity"] == "medium", meta["severity"]

    # And the healthy half: a measured run still publishes its numbers.
    healthy = build_explanation(
        "fairness_report",
        {"fairness_score": 92, "n_passed": 5, "n_total": 5, "metrics": [{"passed": True}] * 5},
    ).metadata()
    assert "92" in healthy["finding"], healthy["finding"]
    assert healthy["severity"] == "info"


def test_inject_accessibility_carries_the_refusal_into_the_document():
    """Verified refusal. The injector is the only path from the explanation to
    the artifact, so a refusal it dropped would exist nowhere a reader looks.
    """
    canvas = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect/></svg>'
    refused = build_explanation(
        "fairness_report",
        {
            "not_assessable": True,
            "not_assessable_reason": "nothing was measured.",
            "excluded_groups": ["C (n=10)"],
        },
    )

    out = inject_accessibility(canvas, refused)

    assert 'role="img"' in out
    assert "COULD NOT CHECK" in desc(out), desc(out)
    assert "C (n=10)" in metadata(out)["finding"]
    assert metadata(out)["severity"] == "medium"

    # A document with no root element is returned unchanged rather than being
    # wrapped in a fabricated one: there is nothing to describe.
    assert inject_accessibility("", refused) == ""
    assert inject_accessibility("not markup", refused) == "not markup"

    # The healthy half: a measured finding is written out as it stands.
    measured = build_explanation(
        "fairness_report",
        {"fairness_score": 92, "n_passed": 5, "n_total": 5, "metrics": [{"passed": True}] * 5},
    )
    assert "92" in desc(inject_accessibility(canvas, measured))


def test_apply_skin_keeps_the_three_badge_states_distinguishable():
    """Verified refusal. A recolour is the last thing to touch the artifact.

    The three states are carried by colour as well as by words, so a palette
    that mapped the could-not-check slate onto the pass green would erase the
    third state after every honest layer above it had done its job. Measured:
    pass #059669 goes to #41ba1b, fail #dc2626 to #b6573a and could-not-check
    #64748b to #5a6a78, three distinct tones, and the badge TEXT survives the
    pass untouched because recolouring is limited to markup.
    """
    after = {}
    for state in (PASS, FAIL, COULD_NOT_CHECK):
        style = _state_style(state)
        skinned = apply_skin(
            f'<svg><rect fill="{style["bg"]}" stroke="{style["color"]}"/>'
            f'<text fill="{style["color"]}">{style["status"]}</text></svg>'
        )
        colour = re.search(r'stroke="(#[0-9a-f]{6})"', skinned).group(1)
        after[state] = colour
        assert style["status"] in skinned, (
            f"the {state} badge lost its label to the recolour: {skinned}"
        )

    assert len(set(after.values())) == 3, (
        f"two of the three states are the same colour after the skin: {after}"
    )
    assert after[COULD_NOT_CHECK] != after[PASS], (
        f"could-not-check is painted the colour of a measured pass: {after}"
    )


def test_apply_skin_leaves_the_not_assessable_banner_readable():
    """The shipped artifact is the skinned one, so the wording is read AFTER it.

    Executed through the real adapter: an empty report renders a canvas whose
    NOT ASSESSABLE wording and slate third-state tone both survive Blanco, and
    applying the skin again changes nothing.

    WHAT THIS DELIBERATELY DOES NOT ASSERT. A first draft banned the Blanco pass
    green #41ba1b from the whole document and went red on two counts that are
    not verdicts: every occurrence in metrics_bar_chart is a ``stop-color`` in
    the validant brand gradient, and effect_sizes paints the word NEGLIGIBLE
    green as an axis LEGEND at the zero line. Both are furniture, the same class
    of mistake as reading a matplotlib tick label as drawn data.
    """
    svg = _render("metrics_bar_chart_to_svg", {})

    joined = " ".join(visible_text(svg))
    assert NOT_ASSESSABLE_TEXT in joined, joined[:300]
    assert svg == apply_skin(svg), "the shipped artifact is already Blanco"

    # The colour of the words themselves, read off the two <text> nodes that
    # carry the banner, rather than looked for anywhere in the document: a
    # document-wide search for the Blanco neutral #5a6a78 passed even when
    # _UNKNOWN_COLOR was sabotaged to the pass green, because other slate tones
    # map onto it. #41ba1b is Blanco's pass green and #b6573a its one warn hue;
    # the third state may borrow neither.
    banner_fills = re.findall(
        r'<text\b[^>]*fill="(#[0-9a-f]{6})"[^>]*>[^<]*' + NOT_ASSESSABLE_TEXT, svg
    )
    assert banner_fills, "the banner carries no explicit fill, so its state is unreadable"
    assert set(banner_fills) == {"#5a6a78"}, (
        f"the could-not-check banner is painted in a measured state's tone: {banner_fills}"
    )
