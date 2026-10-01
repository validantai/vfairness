"""Audit 6, wave 2: the two CRITICAL findings, both a falsy-coalescing neutral.

Both defects are ``X or <neutral>`` on a report surface, and both fire not only
when the value is MISSING but also when it is a genuinely MEASURED ZERO, so a
real measurement of zero became indistinguishable from an absence and was then
printed as a verdict. Three states, never two.

Findings, each REPRODUCED BY EXECUTION on 2026-09-09 before the fix:

- C-1  rendering/adapters_ranking.py:404, an INVERTED verdict.
       ``max(measured_exposures) / (min(measured_exposures) or 1)``. A group
       whose exposure is a MEASURED 0.0 is shut out of the ranking entirely,
       which ``_num`` calls "the single worst reading that panel can produce",
       and the ``or 1`` handed it a denominator of 1. Executed on this repo,
       one metric row and three groups:

         {A: 1.8, B: 1.2, C: 0.0}  ->  "Measured exposure is well-distributed
                                        across the groups shown."
         {A: 1.8, B: 1.2, C: 0.5}  ->  "Large exposure gap between groups,
                                        consider a re-ranking strategy."

       The MORE severe input produced the MORE reassuring sentence. The ratio
       against a shut-out group is unbounded, not 1.8.

- C-2  operations/reporting/compliance.py:2100-2101, a FABRICATED statistic in
       a regulatory narrative.
       ``f"95% CI {round((v.get('ciLow') or 0) * 100)}-..."``. The producers set
       ciLow/ciHigh to None ON PURPOSE when the interval is not computable
       (orchestrator.py:1230), and two other consumers of the same two fields
       already guard with an isinstance check (orchestrator.py:3288 and
       :5847-5852). This third one coalesced. Executed on this repo, row
       {gap 0.23, significant True, ciLow None, ciHigh None}:

         "Black" is selected 23 pts less than "White" on race
         (significant; 95% CI 0-0 pts). Assessed under EU/UK/CH ...

       An infinitely tight interval around zero, printed beside a 23-point point
       estimate. A NaN pair was worse still: ``round(nan * 100)`` raised
       ValueError, the section's ``except Exception`` swallowed it, and every
       disparate-impact finding in the run left the verdict silently.

Every pin below comes in two halves: a REFUSAL pin (the unmeasured or
shut-out state is named, never rounded into a clean-looking number) and an
OVER-CORRECTION CONTROL (the measured path still prints its real answer),
because a fix that refuses everything is as useless as one that passed
everything.
"""

from __future__ import annotations

import html
import re

import pytest

from vfairness.operations.reporting.compliance import build_assurance_verdict
from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg

# ---------------------------------------------------------------------------
# C-1: a MEASURED exposure of 0.0 is an unbounded ratio, not a denominator of 1
# ---------------------------------------------------------------------------

_TEXT_RE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.DOTALL)

# One measured metric, so the page is a real assessment rather than the
# could-not-check canvas: the exposure sentences are what these tests read.
_METRICS = [{"metric_name": "exposure_parity_ratio", "value": 0.62, "threshold": 0.80}]

_ALL_CLEAR = "Measured exposure is well-distributed"
_WIDE_GAP = "Large exposure gap between groups"
_SHUT_OUT = "received ZERO measured exposure"
_UNBOUNDED = "the exposure ratio against them is unbounded"


def _canvas_text(svg: str) -> str:
    """Every ``<text>`` on the canvas, tags stripped and entities decoded."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", body)) for body in _TEXT_RE.findall(svg))


class TestMeasuredZeroExposureIsNotADenominatorOfOne:
    def test_a_shut_out_group_never_earns_the_all_clear(self):
        """THE REFUSAL PIN, and the inversion itself. A measured exposure of 0.0
        used to divide through ``or 1`` and print the panel's all-clear."""
        svg = ranking_fairness_to_svg(
            _METRICS,
            {
                "A": {"mean_exposure": 1.8},
                "B": {"mean_exposure": 1.2},
                "C": {"mean_exposure": 0.0},
            },
        )
        text = _canvas_text(svg)

        assert _ALL_CLEAR not in text, "a group shut out of the ranking is not well-distributed"
        assert _SHUT_OUT in text
        assert "(C)" in text, "the shut-out group has to be named, not just counted"
        assert _UNBOUNDED in text

    def test_the_more_severe_input_is_never_the_more_reassuring_page(self):
        """MONOTONICITY. C=0.0 is strictly worse than C=0.5, so the page for it
        cannot be the calmer of the two. This is the shape the defect had: the
        worse input produced the all-clear and the milder one produced a
        finding."""
        worse = _canvas_text(
            ranking_fairness_to_svg(
                _METRICS,
                {
                    "A": {"mean_exposure": 1.8},
                    "B": {"mean_exposure": 1.2},
                    "C": {"mean_exposure": 0.0},
                },
            )
        )
        milder = _canvas_text(
            ranking_fairness_to_svg(
                _METRICS,
                {
                    "A": {"mean_exposure": 1.8},
                    "B": {"mean_exposure": 1.2},
                    "C": {"mean_exposure": 0.5},
                },
            )
        )

        assert _WIDE_GAP in milder, "the 3.6x spread must still raise its finding"
        assert _ALL_CLEAR not in milder
        # The worse page carries an exposure finding of its own, and the
        # STRONGEST one, rather than the milder page's sentence or none at all.
        assert _SHUT_OUT in worse
        assert _ALL_CLEAR not in worse

    def test_a_measured_zero_still_draws_its_own_number(self):
        """OVER-CORRECTION CONTROL. 0.000 is the finding. Withholding it would
        be the same fabrication pointed the other way, and it is what
        ``_num``'s three-state gate exists to prevent."""
        svg = ranking_fairness_to_svg(
            _METRICS, {"A": {"mean_exposure": 1.8}, "C": {"mean_exposure": 0.0, "count": 0}}
        )
        text = _canvas_text(svg)

        assert "0.000" in text
        assert "n=0" in text
        assert "no exposure reported" not in text, "0.0 was reported; it is not an absence"

    def test_a_real_gap_between_measured_groups_still_reports_its_spread(self):
        """OVER-CORRECTION CONTROL. 1.8 / 0.5 = 3.6, over the 2.0 bar: the
        ordinary wide-gap finding must survive the fix untouched."""
        svg = ranking_fairness_to_svg(
            _METRICS,
            {
                "A": {"mean_exposure": 1.8},
                "B": {"mean_exposure": 1.2},
                "C": {"mean_exposure": 0.5},
            },
        )
        text = _canvas_text(svg)

        assert _WIDE_GAP in text
        assert _SHUT_OUT not in text, "nothing here was shut out"

    def test_a_narrow_measured_spread_still_earns_the_all_clear(self):
        """OVER-CORRECTION CONTROL, the other end. 1.0 / 0.8 = 1.25 is a
        well-distributed panel and must still read as one: a fix that refuses
        every page is as useless as one that passed every page."""
        svg = ranking_fairness_to_svg(
            _METRICS,
            {
                "A": {"mean_exposure": 1.0},
                "B": {"mean_exposure": 0.9},
                "C": {"mean_exposure": 0.8},
            },
        )
        text = _canvas_text(svg)

        assert _ALL_CLEAR in text
        assert _WIDE_GAP not in text
        assert _SHUT_OUT not in text

    def test_the_unmeasured_gate_above_is_untouched(self):
        """The measured half of that line was already correct: a group that
        reported NO exposure enters neither side of the ratio, is named as
        unreported, and is not a shut-out group. Both halves now have a pin."""
        svg = ranking_fairness_to_svg(_METRICS, {"Big": {"mean_exposure": 3.0}, "Bare": {}})
        text = _canvas_text(svg)

        assert "1 group(s) reported no exposure" in text
        assert _SHUT_OUT not in text, "an absent exposure is not a measured zero"
        assert _WIDE_GAP not in text, "no gap may be taken against a number nobody measured"

    def test_every_measured_exposure_zero_claims_no_ratio_at_all(self):
        """The third state on the ratio itself. When every measured exposure is
        0.0 the ratio is 0/0 and does not exist, so the panel must not claim an
        unbounded one either: that would be a second invented number beside the
        first."""
        svg = ranking_fairness_to_svg(
            _METRICS, {"A": {"mean_exposure": 0.0}, "B": {"mean_exposure": 0.0}}
        )
        text = _canvas_text(svg)

        assert _SHUT_OUT in text
        assert "no ratio can be taken" in text
        assert _UNBOUNDED not in text
        assert _ALL_CLEAR not in text

    @pytest.mark.parametrize(
        "group_metrics",
        [
            {"A": {"mean_exposure": 0.0}},
            {"A": {"mean_exposure": 0.0}, "B": {}},
            {"A": {"mean_exposure": 0.0}, "B": {"mean_exposure": 2.0}},
            {"A": {"mean_exposure": 0.0}, "B": {"mean_exposure": 0.0}},
        ],
    )
    def test_a_zero_exposure_never_collapses_the_render(self, group_metrics):
        """The render is an export: a division that raises would leave a caller
        with an empty string and no page at all."""
        svg = ranking_fairness_to_svg(_METRICS, group_metrics)

        assert svg.startswith("<svg") or "<svg" in svg
        assert _ALL_CLEAR not in _canvas_text(svg)


# ---------------------------------------------------------------------------
# C-2: an interval nobody computed is named, never rounded to "95% CI 0-0 pts"
# ---------------------------------------------------------------------------

_FABRICATED_CI = "95% CI 0-0 pts"
_NO_CI = "no 95% confidence interval was computed"

_ROW = {
    "assessable": True,
    "attribute": "race",
    "worstGroup": "Black",
    "referenceGroup": "White",
    "gap": 0.23,
    "significant": True,
}


def _disparate_impact(**row_overrides) -> dict:
    """The one disparate-impact finding the real ``build_assurance_verdict``
    produces for a single per-variable row."""
    row = dict(_ROW, **row_overrides)
    verdict = build_assurance_verdict(per_variable=[row], jurisdiction="EU", domain="hiring")
    hits = [f for f in verdict["findings"] if f["type"] == "disparate_impact"]
    assert len(hits) == 1, f"expected exactly one disparate_impact finding, got {hits}"
    return hits[0]


class TestAnUncomputedConfidenceIntervalIsNeverPrintedAsMeasured:
    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"ciLow": None, "ciHigh": None}, id="explicit-None"),
            pytest.param({}, id="keys-absent"),
            pytest.param({"ciLow": 0.14, "ciHigh": None}, id="only-the-low-bound"),
            pytest.param({"ciLow": None, "ciHigh": 0.31}, id="only-the-high-bound"),
            pytest.param({"ciLow": "n/a", "ciHigh": "n/a"}, id="free-text"),
            pytest.param({"ciLow": True, "ciHigh": True}, id="bool-is-not-a-measurement"),
        ],
    )
    def test_the_uncomputed_interval_is_named_not_rounded_to_zero(self, overrides):
        """THE REFUSAL PIN. "95% CI 0-0 pts" is an infinitely tight interval
        around zero printed beside a 23-point gap: both invented and self
        contradictory, in the one sentence a compliance reader relies on."""
        finding = _disparate_impact(**overrides)

        assert _FABRICATED_CI not in finding["evidence"]
        assert _NO_CI in finding["evidence"]
        assert "could not be checked" in finding["evidence"]

    def test_a_nan_interval_no_longer_deletes_the_whole_section(self):
        """The sibling defect on the same line. ``round(nan * 100)`` raised
        ValueError, the section's ``except Exception`` swallowed it, and the
        23-point gap itself vanished from the verdict with no trace but a debug
        log line."""
        finding = _disparate_impact(ciLow=float("nan"), ciHigh=float("nan"))

        assert _NO_CI in finding["evidence"]
        assert "23 pts less" in finding["evidence"], "the gap must survive its own CI"

    def test_the_severity_and_the_gap_are_untouched_by_a_missing_interval(self):
        """A withheld interval withdraws the PRECISION claim, never the finding:
        the gap is measured, significant and still critical."""
        without = _disparate_impact(ciLow=None, ciHigh=None)
        with_ci = _disparate_impact(ciLow=0.14, ciHigh=0.31)

        assert without["severity"] == with_ci["severity"] == "critical"
        assert without["attribute"] == with_ci["attribute"] == "race"
        assert "23 pts less" in without["evidence"]

    def test_a_real_interval_still_prints_its_real_numbers(self):
        """OVER-CORRECTION CONTROL. 0.14 to 0.31 must read as 14-31 pts."""
        finding = _disparate_impact(ciLow=0.14, ciHigh=0.31)

        assert "95% CI 14-31 pts" in finding["evidence"]
        assert _NO_CI not in finding["evidence"]

    def test_a_measured_lower_bound_of_zero_is_still_a_number(self):
        """OVER-CORRECTION CONTROL, and the defect class itself. An interval
        that legitimately spans zero (ciLow 0.0) is a MEASUREMENT: the old
        ``or 0`` could not tell it from an absence, and the new guard must not
        withhold it."""
        finding = _disparate_impact(ciLow=0.0, ciHigh=0.18, significant=False)

        assert "95% CI 0-18 pts" in finding["evidence"]
        assert _NO_CI not in finding["evidence"]

    def test_a_measured_zero_width_interval_is_still_printed(self):
        """OVER-CORRECTION CONTROL at the exact collision point: a measured
        0.0-to-0.0 interval prints the same characters the fabrication used to
        print, and that is correct, because here they were measured."""
        finding = _disparate_impact(ciLow=0.0, ciHigh=0.0, significant=False)

        assert _FABRICATED_CI in finding["evidence"]
        assert _NO_CI not in finding["evidence"]
