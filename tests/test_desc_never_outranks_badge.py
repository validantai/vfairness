"""The accessible description may never outrank the badge.

Every SVG this library renders carries an accessible ``<desc>`` built by
``rendering.explain.build_explanation``. For a sighted reader the ``<desc>`` is
invisible and the badge is the chart; for a screen-reader user the ``<desc>`` IS
the chart, and for a report pipeline or an assistant parsing the file it is the
only text that is machine-readable. So the two must state the same one of the
three states, and where they differ the description must be the LESS confident
of the pair. It may qualify a verdict the canvas made. It may never make one the
canvas withheld.

The defect this file pins was exactly that inversion.
``experiment_recommendation_to_svg({"decision": "ship it now"})`` drew a canvas
that named no recommendation this library grades, while the ``<desc>`` read
"Recommendation: Ship It Now (confidence %)". A reader who cannot see the canvas
was handed a recommendation nobody made, in the one sentence they get. The same
shape sat in the ranking and regression findings: their "; all pass" clause was
written from the badge list, which the adapters TRUNCATE
(``badges[:6]`` / ``badges[:4]``) while the pass counts are taken over all of
them, so seven ranking metrics with the only failure in seventh place produced
"6/7 metrics pass ...; all pass" underneath an UNFAIR headline.

This is a GENERAL guard, not three regression pins. It renders a representative
set of adapters in their could-not-check state and asserts that no confident
verdict phrase reaches the description. Adding a chart that fabricates a verdict
for an unmeasured run should fail here without anyone remembering to add a case,
which is why the vocabulary is shared and the canvas is checked as well as the
text.
"""

from __future__ import annotations

import html
import re
import warnings

import pytest

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)

import vfairness.rendering as R  # noqa: E402

_DESC_RE = re.compile(r"<desc>(.*?)</desc>", re.S)
_DESC_TAG_RE = re.compile(r"<desc>.*?</desc>", re.S)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
_METADATA_RE = re.compile(r"<metadata.*?</metadata>", re.S)
_SEVERITY_TAIL_RE = re.compile(r"\s*\(severity: [A-Za-z]+\)\s*$")


def _desc(svg: str) -> str:
    """The accessible description, unescaped, as a screen reader receives it."""
    m = _DESC_RE.search(svg)
    assert m, "every rendered chart must carry an accessible <desc>"
    return html.unescape(m.group(1))


def _finding(svg: str) -> str:
    """The claim inside the description: its chart name and severity removed.

    ``ChartExplanation.caption`` builds "<title>: <finding> (severity: X)", and
    the title is a chart NAME, not a claim. Scanning the whole string would trip
    over "Experiment Recommendation: ..." and report the name of the chart as a
    recommendation, which is the mirror image of the bug this file is about.
    """
    desc = _desc(svg)
    m = _TITLE_RE.search(svg)
    if m:
        prefix = html.unescape(m.group(1)) + ": "
        if desc.startswith(prefix):
            desc = desc[len(prefix) :]
    return _SEVERITY_TAIL_RE.sub("", desc)


def _canvas(svg: str) -> str:
    """The drawn canvas: the markup a sighted reader sees, minus the a11y layer."""
    return _METADATA_RE.sub("", _DESC_TAG_RE.sub("", svg)).upper()


# What a canvas says when it withholds its verdict. One of these must be on the
# canvas for a case below to count as could-not-check; without that check the
# whole file could pass against charts that are quietly asserting a verdict on
# both surfaces.
WITHHELD_MARKS = (
    "COULD NOT CHECK",
    "NOT ASSESSABLE",
    "NOT ASSESSED",
    "NOT CHECKED",
    "NOT SCORED",
    "NO RECOMMENDATION MADE",
    "UNKNOWN",
)

# Phrases that assert a measured outcome. None of them may appear in the
# description of a chart whose canvas withheld its verdict: each one is a claim
# that something was compared to a threshold, graded, or recommended.
CONFIDENT_VERDICTS = (
    "all pass",
    "metrics pass",
    "metric(s) passed",
    "recommendation:",
    "approved",
    "well calibrated",
    "is feasible",
    "not significant",
    "drift not detected",
    "overall fairness is",
    "constraint satisfied",
    "pipeline passed",
    "is fair",
    "no disparity detected",
)


def _verdicts_in(text: str) -> list:
    low = text.lower()
    return [phrase for phrase in CONFIDENT_VERDICTS if phrase in low]


# A representative slice of the adapter surface, each called so that it has
# nothing to grade. Deliberately called through the public
# ``vfairness.rendering`` names, with the arguments a caller would actually
# pass, so this exercises the shipped entry points rather than a finder.
WITHHELD_CASES = {
    "ranking_fairness": lambda: R.ranking_fairness_to_svg([], {}),
    "regression_fairness": lambda: R.regression_fairness_to_svg({}, {}, []),
    "experiment_recommendation": lambda: R.experiment_recommendation_to_svg({}),
    "experiment_results": lambda: R.experiment_results_to_svg({}),
    "power_analysis": lambda: R.power_analysis_to_svg([]),
    "threshold_optimization": lambda: R.threshold_optimization_to_svg({}),
    "reweighting_comparison": lambda: R.reweighting_comparison_to_svg({}),
    "fairness_detailed_report": lambda: R.fairness_detailed_report_to_svg({}),
    "auto_discovery": lambda: R.auto_discovery_to_svg(),
    "data_validation": lambda: R.data_validation_to_svg({}),
}


def _render(thunk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return thunk()


@pytest.mark.parametrize("chart", sorted(WITHHELD_CASES))
def test_the_canvas_of_each_case_really_does_withhold_its_verdict(chart):
    """Keeps the guard below honest: a case that grades something proves nothing."""
    svg = _render(WITHHELD_CASES[chart])
    canvas = _canvas(svg)
    assert any(mark in canvas for mark in WITHHELD_MARKS), (
        f"{chart} was chosen as a could-not-check case but its canvas states no "
        f"withheld verdict, so the description assertion would be vacuous"
    )


@pytest.mark.parametrize("chart", sorted(WITHHELD_CASES))
def test_the_description_claims_no_verdict_the_canvas_withheld(chart):
    svg = _render(WITHHELD_CASES[chart])
    found = _verdicts_in(_finding(svg))
    assert not found, (
        f"{chart}: the canvas withheld its verdict, but the accessible <desc> "
        f"claims {found}. A screen-reader user gets this sentence and nothing "
        f"else. Description: {_desc(svg)!r}"
    )


def test_a_decision_this_library_cannot_make_is_not_read_out_as_a_recommendation():
    """The original inversion: an unrecognised decision string, echoed verbatim.

    ``RecommendationDecision`` is a closed vocabulary of four. Anything else is
    not a decision this library made, and the canvas does not name it, so
    neither may the description.
    """
    svg = _render(lambda: R.experiment_recommendation_to_svg({"decision": "ship it now"}))
    desc = _finding(svg).lower()
    assert "ship it now" not in desc
    assert "recommendation:" not in desc
    assert "could not check" in desc


def test_all_pass_is_not_written_when_the_counts_say_a_metric_did_not_pass():
    """The truncated-badge inversion, on the counts the canvas itself grades."""
    results = [
        {"metric_name": "ndkl", "value": 0.02, "threshold": 0.1, "is_fair": True} for _ in range(6)
    ]
    results.append({"metric_name": "ndkl", "value": 0.9, "threshold": 0.1, "is_fair": False})
    svg = _render(lambda: R.ranking_fairness_to_svg(results, {"a": {"mean_exposure": 0.5}}))
    desc = _desc(svg)
    assert "6/7 metrics pass" in desc
    assert "all pass" not in desc.lower()


# ── The guard must be able to fail ──────────────────────────────────────────
# A forbidden-phrase list proves nothing unless the phrases are ones this
# library really writes when it HAS measured something. These controls render
# the same two charts with real input and assert the verdict is still stated:
# without them, deleting every description would satisfy every test above.


def test_a_graded_recommendation_is_still_read_out():
    svg = _render(
        lambda: R.experiment_recommendation_to_svg(
            {"decision": "deploy_treatment", "confidence": 0.83}
        )
    )
    desc = _desc(svg)
    assert "Recommendation: Deploy Treatment (confidence 83%)" in desc
    assert "COULD NOT CHECK" not in desc


def test_a_fully_passing_ranking_report_still_says_all_pass():
    results = [
        {"metric_name": "ndkl", "value": 0.02, "threshold": 0.1, "is_fair": True},
        {
            "metric_name": "exposure_parity_difference",
            "value": 0.01,
            "threshold": 0.1,
            "is_fair": True,
        },
    ]
    svg = _render(
        lambda: R.ranking_fairness_to_svg(
            results, {"a": {"mean_exposure": 0.5}, "b": {"mean_exposure": 0.45}}
        )
    )
    desc = _desc(svg)
    assert "2/2 metrics pass" in desc
    assert "all pass" in desc
    assert _verdicts_in(desc), "the forbidden vocabulary must match real library output"


# ── The same rule, applied to the FILE an export path leaves behind ─────────
# A chart that could not be drawn must not leave the previous chart sitting at
# save_path, and must not leave a zero-byte file either. Both are more
# confident than the run: the stale one asserts the last run's verdict as this
# run's, and the empty one cannot even be read as could-not-check, because the
# reader sees a broken image and cannot tell a failed render from a failed
# model. ``open(save_path, "w")`` truncates BEFORE it writes, so a failure
# after that point produced exactly the zero-byte case, silently.

_EXPORTS = {
    "threshold_optimization": (
        lambda **kw: R.threshold_optimization_to_svg(
            {"constraint_type": "demographic_parity"}, **kw
        )
    ),
    "reweighting_comparison": lambda **kw: R.reweighting_comparison_to_svg(
        {"method_results": []}, **kw
    ),
}


@pytest.mark.parametrize("chart", sorted(_EXPORTS))
def test_an_export_path_never_leaves_a_stale_chart_when_it_cannot_render(chart, tmp_path):
    from vfairness.rendering import adapters_post_processing as app

    out = tmp_path / f"{chart}.svg"
    good = _render(lambda: _EXPORTS[chart](save_path=str(out)))
    assert "This report was not drawn" not in good, "the first render must be a real chart"
    stale = out.read_text()

    original = app.JINJA2_AVAILABLE
    try:
        app.JINJA2_AVAILABLE = False
        # UPDATED 2026-08-28: the call now RAISES instead of warning and
        # returning "". The file invariants below are unchanged and are still
        # the point of this test: the export must be written BEFORE the raise,
        # because an exception alone leaves the stale chart in place for a
        # pipeline that catches per report and carries on.
        with pytest.raises(ImportError, match=r"vfairness\[rendering\]"):
            _EXPORTS[chart](save_path=str(out))
    finally:
        app.JINJA2_AVAILABLE = original

    written = out.read_text()
    assert written != stale, "the chart that was never drawn left the previous chart in place"
    assert written.strip(), "an export path must never leave a zero-byte file"
    assert "COULD NOT CHECK" in written


@pytest.mark.parametrize("chart", sorted(_EXPORTS))
def test_a_failed_render_writes_what_it_returns(chart, tmp_path):
    from vfairness.rendering import adapters_post_processing as app

    out = tmp_path / f"{chart}.svg"

    def boom(*a, **k):
        raise RuntimeError("template exploded")

    original = app.render_svg
    try:
        app.render_svg = boom
        with pytest.warns(UserWarning, match="SVG rendering failed"):
            returned = _EXPORTS[chart](save_path=str(out))
    finally:
        app.render_svg = original

    assert returned.strip(), "a failed render must never hand back an empty string"
    assert out.read_text() == returned, (
        "the caller was handed a fallback chart that never reached save_path"
    )
