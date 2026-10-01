"""BGL4 audit of batch A-rendering-1: three overturns, each demonstrated.

These tests were EVIDENCE, not fixes: each one failed against the code as it
stood on 2026-09-27 and named a renderer that batch A-rendering-1 graded PROVEN.
BGL5 closed them the same day, so each is now an ASSERTION OF THE CORRECTED
BEHAVIOUR on the same fixture, with the measured AFTER recorded beside the
measured BEFORE in its docstring. The over-correction controls and the sabotage
evidence live in ``tests/test_bgl5_post_processing_2_and_rendering.py``.

ONE PART IS STILL OPEN and is pinned there as a strict xfail: the temporal
summary card's trend CHIP is drawn by
``src/vfairness/rendering/templates/temporal_analysis.svg``, a file outside the
A-rendering-1 batch, so it is deferred rather than quietly dropped.

Every fixture is built by the library's OWN producer
(``FairnessMonitor`` / ``TemporalFairnessAnalyzer``), so none of them depends on
a hand-made duck type: the objection "no real caller produces that" does not
apply.

What the audited grades rest on is ``tests/test_adapters_zero_and_empty.py``,
which drives each renderer with input that is EMPTY. Every case below supplies
input that is PARTIAL instead: some of the quantity was measured and some of it
was not. That is the commoner shape and the one the harness never tries.

Read on the RENDERED artifact only: the contents of ``<text>`` nodes for the
canvas and the ``<metadata>`` JSON for the machine-readable layer. Never the raw
markup, because a token inside an attribute or a font name is not something a
reader sees ("Verdana" contains "n/a", and an empty axis still carries a "0.0"
tick).
"""

from __future__ import annotations

import json
import re
import warnings

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.operations.monitoring.tracker import (  # noqa: E402
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
)
from vfairness.rendering.adapters_monitoring import (  # noqa: E402
    alert_timeline_to_svg,
    monitoring_dashboard_to_svg,
    temporal_analysis_to_svg,
)

_TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_META = re.compile(r"<metadata[^>]*><!\[CDATA\[(.*?)\]\]></metadata>", re.S)


def canvas(svg: str) -> str:
    """Everything a sighted reader reads, as one string."""
    out = []
    for raw in _TEXT_NODE.findall(svg):
        stripped = _TAG.sub("", raw).strip()
        if stripped:
            out.append(stripped)
    return " ".join(out)


def meta(svg: str) -> dict:
    match = _META.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation block"
    return json.loads(match.group(1))


def _partially_checked_window():
    """One REAL monitoring window: half its metrics checked, one group dropped.

    ``group_region`` has two large groups, so both its metrics are computed and
    both are compared to a guardrail. ``group_gender`` carries an "Other"
    stratum of 7 rows, below the default ``min_samples=30``: the tracker drops
    it, records it in ``excluded_groups``, and leaves both gender metrics NaN
    with no entry in ``alerts`` at all. So this window applied a guardrail to
    two of its four metrics and could not check the other two.
    """
    rng = np.random.default_rng(1)
    n_a, n_b, n_drop = 300, 300, 7
    total = n_a + n_b + n_drop
    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(
                [
                    (rng.random(n_a) < 0.70).astype(int),
                    (rng.random(n_b) < 0.69).astype(int),
                    np.zeros(n_drop, dtype=int),
                ]
            ),
            "label": (rng.random(total) < 0.5).astype(int),
            "group_gender": ["Male"] * n_a + ["Female"] * n_b + ["Other"] * n_drop,
            "group_region": ["North"] * (total // 2) + ["South"] * (total - total // 2),
        }
    )
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = monitor.update_and_check(frame)
    # The premise, asserted rather than assumed.
    assert window.excluded_groups == {"group_gender": {"Other": 7}}, window.excluded_groups
    assert len(window.metrics) == 4 and len(window.alerts) == 2, (window.metrics, window.alerts)
    return window


def _analyzer(values):
    """A REAL TemporalFairnessAnalyzer holding one metric over len(values) days."""
    analyzer = TemporalFairnessAnalyzer()
    base = pd.Timestamp("2026-01-01")
    for offset, value in enumerate(values):
        analyzer.update_daily_metrics(
            base + pd.Timedelta(days=offset), {"demographic_parity_difference": value}
        )
    return analyzer


# The shipped artifact is already Blanco, so the card's #059669 has become the
# Blanco pass green by the time anyone reads the SVG. Asserting the pre-skin hex
# is what a first draft of this test did, and it went red on the CONTROL.
_BLANCO_PASS = "#41ba1b"

_CHIP = re.compile(
    r'<text\b[^>]*font-size="7\.5"[^>]*fill="(#[0-9a-fA-F]{6})"[^>]*>'
    r"(STABLE|INCREASING|DECREASING|NOT TRACKED)</text>"
)


def _card_chip(svg: str):
    """The (fill, label) of the metric summary card's trend chip.

    Read off the ONE text node the template draws it in, not searched for
    anywhere in the document: a document-wide search for a colour matches other
    slate and green tones, and a search for the word matches the degradation
    table's own chip further down the page.
    """
    found = _CHIP.findall(svg)
    assert found, "the summary card carries no trend chip, so its state is unreadable"
    return found[0]


def test_alert_timeline_calls_a_half_checked_window_clean():
    """OVERTURNS the PROVEN grade on ``alert_timeline_to_svg``.

    ``checked = len(alerts) > 0`` is a two-state test at WINDOW level over a
    per-metric record. A window that compared two of its four metrics to a
    guardrail, and could not check the other two, is therefore counted as fully
    clean: the canvas prints "CLEAN 1" with an OK badge and the accessible layer
    publishes "1 clean" at severity info, the same artifact a window that really
    was watched end to end produces.

    ``monitoring_dashboard_to_svg`` reads the SAME object and gets this right:
    "2 of 4 metrics checked against a guardrail" on the canvas, "2 metric(s)
    were never compared to a guardrail and are not covered by this finding" in
    the <desc>, severity floored to medium. The timeline carries neither.

    AFTER (BGL5): the timeline makes the same per-metric read, so this window is
    neither alerted nor clean. Canvas "1 monitoring events / 0 with alerts / 0
    clean", CLEAN 0, the row's ALERTS cell "2 of 4 checked", its FLAGGED cell "2
    metric(s) never checked" and its STATUS "PARTIAL" with no emerald pill; the
    published finding adds "2 metric(s) across 1 partly checked window(s) were
    never compared to a guardrail and are not covered by this finding." at
    severity medium. A window every one of whose metrics WAS checked still reads
    CLEAN 1 / OK / info, which the BGL5 control asserts.
    """
    window = _partially_checked_window()

    # CONTROL, so the bar below cannot be dismissed as unreachable: the sibling
    # renderer clears it on the very same object.
    dashboard = meta(monitoring_dashboard_to_svg(window))
    assert dashboard["severity"] == "medium", dashboard["severity"]
    assert "not covered by this finding" in str(dashboard["finding"]), dashboard["finding"]

    svg = alert_timeline_to_svg([window])
    published = meta(svg)
    assert published["severity"] != "info", (
        "a window with two unchecked metrics is graded the severity of a clean "
        f"run: {published['severity']} / {published['finding']}"
    )
    assert "not covered by this finding" in str(published["finding"]), (
        "the timeline states a clean count with no caveat for the metrics it "
        f"never checked: {published['finding']}"
    )
    assert "CLEAN 1" not in canvas(svg), (
        "a half-checked window is tallied as clean on the canvas: " + canvas(svg)[:300]
    )


def test_neither_monitoring_renderer_names_the_group_the_tracker_dropped():
    """OVERTURNS the PROVEN grades on the two monitoring window renderers.

    ``WindowMetrics.excluded_groups`` is the tracker's record of the strata that
    ``min_samples`` kept OUT of every rate on the page, and
    ``_group_positive_rates`` says so in its own docstring: "Groups below
    min_samples are absent from the result. That absence is invisible to a
    caller looking only at this dict." No renderer reads the key (grep
    ``excluded_groups`` under ``src/vfairness/rendering``: adapters_fairness and
    adapters.py only), so the Group Positive Rates table presents the surviving
    groups as though they were the groups.

    This is the same defect the same campaign fixed in ``build_explanation`` for
    the six adapters_fairness charts, in the renderer family next door.

    AFTER (BGL5): both renderers read the key through
    ``adapters_monitoring._window_excluded_groups``. The dashboard draws "Other
    (n=7, group_gender) / not measured" as a row IN the Group Positive Rates
    table, ahead of the measured rows because that list is truncated at eight, and
    both charts pass ``excluded_groups`` on the data dict so
    ``explain._excluded_clause`` appends "Excluded: Other (n=7, group_gender)." to
    the finding and floors the severity with it.
    """
    window = _partially_checked_window()

    for name, svg in (
        ("monitoring_dashboard", monitoring_dashboard_to_svg(window)),
        ("alert_timeline", alert_timeline_to_svg([window])),
    ):
        everything = canvas(svg) + " " + json.dumps(meta(svg))
        assert "Other" in everything, (
            f"{name}: the 'Other' stratum was dropped from every rate on this "
            f"page and is named nowhere a reader looks"
        )


def test_temporal_card_paints_an_unfitted_trend_as_stable():
    """OVERTURNS the PROVEN grade on ``temporal_analysis_to_svg``.

    ``detect_trend`` answers ``("not_assessed", nan)`` for a metric the analyzer
    HOLDS but cannot fit a line through, and warns while doing it. The trends
    TABLE renders that honestly ("slope: N/A/day", "not_assessed") and the
    counts exclude it, which is the fix the adapter's own comment describes. The
    metric summary CARD, the largest element on the canvas, still reads the
    statistic straight off ``get_metric_summary``: ``trend_direction`` is
    "not_assessed", which falls into the template's ``{% else %}`` arm, so the
    chip reads STABLE in the pass green #059669, and ``trend_slope`` is NaN,
    which is not None, so "Slope/day" prints "+nan".

    STABLE is the calmest reading this card has and no trend was fitted.

    AFTER (BGL5), PARTLY. ``adapters_monitoring.temporal_analysis_to_svg`` now
    makes the same fitted/not-fitted test for the CARD that it already made for
    the trends table, so "Slope/day" prints "not measured" in the neutral slate,
    the range bar is slate rather than green, and the card's data dict carries
    ``trend_fitted`` False with the label "NOT FITTED" and the slate colour ready
    for the template.

    THE CHIP ITSELF IS STILL OPEN. Its label and colour are computed inside
    ``templates/temporal_analysis.svg``, whose ``{% else %}`` arm maps every
    direction other than increasing/decreasing to STABLE in the pass green, and
    that file is outside the A-rendering-1 batch. It is pinned as a strict xfail
    in ``tests/test_bgl5_post_processing_2_and_rendering.py::
    test_the_temporal_card_chip_still_certifies_an_unfitted_trend_as_stable``, so
    it turns red the moment the template is fixed. What this test asserts is the
    half that landed, plus the control below, unchanged.
    """
    unfitted = _analyzer([0.05, 0.06, 0.07])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = unfitted.get_metric_summary("demographic_parity_difference")
        svg = temporal_analysis_to_svg(unfitted)

    # The premise, asserted rather than assumed.
    assert summary["trend_direction"] == "not_assessed", summary
    assert summary["trend_slope"] != summary["trend_slope"], summary  # NaN

    # CONTROL, so this is not merely a ban on the word: a metric with enough
    # history to be fitted and genuinely flat SHOULD get the green STABLE chip,
    # and does.
    flat = _analyzer([0.05] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat_svg = temporal_analysis_to_svg(flat)
    assert _card_chip(flat_svg) == (_BLANCO_PASS, "STABLE")
    assert "Slope/day +0.0000" in canvas(flat_svg), canvas(flat_svg)[:280]

    assert "nan" not in canvas(svg), (
        "the NaN sentinel detect_trend writes for 'not fitted' is drawn as a "
        "measured slope: " + canvas(svg)[:280]
    )
    assert "Slope/day not measured" in canvas(svg), canvas(svg)[:280]
