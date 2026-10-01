"""The one renderer on the public surface that no test called.

Measured 2026-09-25: of the 45 `*_to_svg` renderers, 44 are exercised by some test
and `FairnessAnalyzer.report_to_svg` was exercised by none. That matters more here
than for most units, because an SVG is an EXPORT: it leaves the building, an auditor
reads it, and it outlives the run that produced it. The one renderer this campaign
did open, `BiasAuditReport.to_svg`, turned out to have three live defects at once and
printed "OVERALL RISK 0% MINIMAL" over an audit that had assessed nothing.

What the four worlds actually render, and it is mostly good news:

  healthy, two groups     FAIRNESS SCORE 100 /100, "5P 0F". A measured verdict.
  one group only          "NOT ASSESSABLE", "could not check: no metric was
                          verified", "5 metric(s) could not be checked".
  two rows                same refusal.
  ONE LABEL ONLY          FAIRNESS SCORE 100 /100 with "1P 0F" and "4 metric(s)
                          could not be checked".

That last row is the interesting one and it is NOT fixed here, deliberately. The
score is computed over the metrics that could be graded, one of five, and the canvas
discloses the partial coverage in three separate places: the pass/fail counts beside
the number, an explicit "4 metric(s) could not be checked" line, and the accessible
description, whose severity is floored for exactly this case (`explain._fr_fairness_report`
appends "metric(s) were never compared to a threshold" and calls `_floor_sev`). A
headline of 100 over a disclosed denominator of 1 of 5 is a PRESENTATION concern, not
a fabricated verdict, and changing it would be the over-correction this file's
neighbours keep warning about. It is recorded here so the next reader sees the
judgement rather than re-deriving it, and so that if the disclosure ever disappears
the test below fails.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

N = 120


def _analyzer(kind: str) -> FairnessAnalyzer:
    rng = np.random.default_rng(3)
    y = (rng.random(N) < 0.5).astype(int)
    pred = y.copy()
    groups = np.array(["a"] * (N // 2) + ["b"] * (N // 2))
    if kind == "single_group":
        groups = np.array(["a"] * N)
    elif kind == "one_label":
        y = np.zeros(N, dtype=int)
        pred = np.zeros(N, dtype=int)
    elif kind == "n_equals_2":
        y, pred, groups = y[:2], pred[:2], np.array(["a", "b"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y, pred, groups)


def _render(kind: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _analyzer(kind).report_to_svg()


def _visible(svg: str) -> str:
    return " | ".join(t.strip() for t in re.findall(r">([^<>]{1,90})<", svg) if t.strip())


@pytest.mark.parametrize("kind", ["single_group", "n_equals_2"])
def test_a_dashboard_over_nothing_measurable_withholds_its_score(kind):
    seen = _visible(_render(kind))
    assert "NOT ASSESSABLE" in seen, f"{kind} rendered a fairness score"
    assert "could not check" in seen.lower()


def test_control_a_real_two_group_dashboard_states_its_score():
    seen = _visible(_render("healthy"))
    assert "NOT ASSESSABLE" not in seen, "a measurable dashboard was withheld"
    assert "FAIRNESS SCORE" in seen
    assert re.search(r"\b100\b", seen), "the measured score disappeared"


def test_a_partial_dashboard_discloses_how_many_metrics_were_not_checked():
    """The judgement recorded in this file's docstring. The headline score is kept,
    and the disclosure beside it is what makes that defensible, so the disclosure is
    what gets pinned."""
    seen = _visible(_render("one_label"))
    assert "could not be checked" in seen, (
        "a dashboard scored from a minority of metrics said nothing about the rest"
    )
    assert re.search(r"4 metric\(s\) could not be checked", seen), (
        f"the unchecked count is no longer named on the canvas: {seen[:200]}"
    )


def test_the_accessible_description_agrees_with_the_canvas():
    """A screen-reader user and a sighted reader must be told the same thing. The
    canvas and the <desc> are produced by different code, and they have disagreed
    before on exactly this class of report."""
    svg = _render("single_group")
    desc = re.search(r"<desc[^>]*>(.*?)</desc>", svg, re.S)
    assert desc, "the export carries no accessible description at all"
    text = desc.group(1).lower()
    assert "could not check" in text or "not assessable" in text, (
        f"the canvas refuses and the description does not: {text[:160]}"
    )


def test_every_world_produces_a_readable_export():
    """The worst of the three states is an artifact that cannot be read as anything.
    `method_comparison_to_svg([])` used to return a zero-length string, so a caller
    passing save_path wrote a 0-byte file and got no error."""
    for kind in ("healthy", "single_group", "one_label", "n_equals_2"):
        svg = _render(kind)
        assert len(svg) > 2000, f"{kind} produced a {len(svg)}-byte export"
        assert svg.lstrip().startswith("<"), f"{kind} produced something that is not markup"
