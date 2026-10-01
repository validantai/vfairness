"""Verdict tests for :mod:`vfairness.rendering.adapters_ranking`.

Closes the coverage half of register finding #22 for the ranking-fairness
adapter, which sat at 16.3 percent of 64 statements.

The page used to print FAIR or UNFAIR from ``n_pass == n_total``. As with the
regression adapter, that comparison is also true when nothing was measured, so
a call carrying group exposures and no metric results rendered a green FAIR out
of 0 == 0. The headline is graded in the adapter now and has three states, so
the nothing-measured case renders NOT ASSESSED and is pinned below.
"""

import re

import pytest

from vfairness.rendering.adapters_ranking import (
    _UNKNOWN as _UNKNOWN_SLATE,
)
from vfairness.rendering.adapters_ranking import (
    _headline,
    _pf,
    _pf_color,
    ranking_fairness_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


GROUP_METRICS = {
    "A": {
        "mean_exposure": 0.59,
        "avg_position": 2.0,
        "median_position": 2,
        "min_position": 0,
        "max_position": 4,
        "count": 5,
    },
    "B": {
        "mean_exposure": 0.32,
        "avg_position": 7.0,
        "median_position": 7,
        "min_position": 5,
        "max_position": 9,
        "count": 5,
    },
}


def _pass_ratio(svg: str) -> str:
    found = re.findall(r">(\d+/\d+)<", svg)
    assert found, "the metrics-pass ratio was not rendered"
    return found[0]


class TestHeadlineVerdict:
    def test_all_metrics_inside_threshold_read_fair(self):
        results = [
            {"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.1},
            {"metric_name": "ndcg_parity", "value": 0.01, "threshold": 0.1},
        ]

        svg = ranking_fairness_to_svg(results, GROUP_METRICS)

        assert ">FAIR<" in svg
        assert ">UNFAIR<" not in svg
        assert _pass_ratio(svg) == "2/2"

    def test_a_failing_metric_flips_the_headline_and_is_named(self):
        results = [
            {"metric_name": "exposure_parity", "value": 0.27, "threshold": 0.1, "is_fair": False},
            {"metric_name": "ndcg_parity", "value": 0.03, "threshold": 0.1, "is_fair": True},
        ]

        svg = ranking_fairness_to_svg(results, GROUP_METRICS)

        assert ">UNFAIR<" in svg
        assert ">FAIR<" not in svg
        assert _pass_ratio(svg) == "1/2"
        assert "EXPOSURE_PARITY" in svg
        assert "1 metric(s) fail threshold" in svg

    def test_the_verdict_follows_is_fair_when_the_caller_supplies_it(self):
        """A metric whose value looks small but which the metric module judged
        unfair must still render FAIL: the adapter must not re-decide it."""
        results = [
            {"metric_name": "exposure_parity", "value": 0.01, "threshold": 0.1, "is_fair": False}
        ]

        svg = ranking_fairness_to_svg(results, GROUP_METRICS)

        assert ">UNFAIR<" in svg
        assert _pass_ratio(svg) == "0/1"

    def test_without_is_fair_the_adapter_falls_back_to_the_threshold_comparison(self):
        results = [{"metric_name": "exposure_parity", "value": 0.27, "threshold": 0.1}]

        svg = ranking_fairness_to_svg(results, GROUP_METRICS)

        assert ">UNFAIR<" in svg

    def test_an_unmeasurable_metric_fails_closed(self):
        results = [{"metric_name": "exposure_parity", "value": float("nan"), "threshold": 0.1}]

        svg = ranking_fairness_to_svg(results, GROUP_METRICS)

        assert ">UNFAIR<" in svg
        assert _pass_ratio(svg) == "0/1"

    @pytest.mark.parametrize(
        ("value", "expected"), [(0.0, "PASS"), (0.1, "PASS"), (0.11, "FAIL"), (1.0, "FAIL")]
    )
    def test_the_pass_rule_is_at_or_below_the_threshold(self, value, expected):
        # REVISED 2026-09-10 (readiness-6): this called ``_pf(value, 0.1)`` with
        # NO metric name, and the adapter quietly substituted the invented name
        # "disparity" so the resolver would answer lower-is-better. The rule
        # under test here is the INCLUSIVE boundary on a violation magnitude
        # (0.1 against a 0.1 bound passes), so it is now stated against a real
        # lower-is-better metric this module actually ships. Same values, same
        # expectations, and no longer resting on a stand-in name.
        assert _pf(value, 0.1, "exposure_parity") == expected
        assert _pf(value, 0.1, "ndkl") == expected

    @pytest.mark.parametrize("value", [0.0, 0.1, 0.11, 1.0, 0.62])
    def test_a_row_with_no_metric_name_is_never_graded(self, value):
        """An empty name is missing EVIDENCE about the direction, not a magnitude.

        The adapter used to hand the shared resolver the invented name
        "disparity" whenever the caller named no metric, so an unnamed row was
        graded as a violation magnitude. Measured on this repo: ``_pf(0.62,
        0.80, "")`` returned "PASS" with the #059669 pass green and the rendered
        SVG carried a green PASS badge over 0.620, while EVERY other
        unresolvable name -- "mystery_metric", and the ``metric_name`` default
        "metric" -- correctly returned NOT CHECKED. 0.62 against 0.80 is the
        exposure-parity shape this module ships, and on that four-fifths floor
        it is a FAIL, so the empty name was the one input that turned a breach
        into an all-clear.
        """
        assert _pf(value, 0.1, "") == "NOT CHECKED"
        assert _pf_color(value, 0.1, "") == _UNKNOWN_SLATE
        assert _pf(value, 0.1) == "NOT CHECKED"

    def test_the_unnamed_row_is_not_green_on_the_rendered_canvas(self):
        """Read the refusal out of the SVG, not only out of the helper."""
        svg = ranking_fairness_to_svg(
            results=[{"metric_name": "", "value": 0.62, "threshold": 0.80}],
            group_metrics=GROUP_METRICS,
        )

        assert ">NOT CHECKED<" in svg
        assert ">FAIR<" not in svg
        assert "#059669" not in svg

    def test_a_named_ratio_metric_still_grades_in_both_directions(self):
        """Does-not-overcorrect: refusing the UNNAMED row must not mute the named ones."""
        assert _pf(0.62, 0.80, "exposure_parity_ratio") == "FAIL"
        assert _pf_color(0.62, 0.80, "exposure_parity_ratio") == "#dc2626"
        assert _pf(0.90, 0.80, "exposure_parity_ratio") == "PASS"
        assert _pf_color(0.90, 0.80, "exposure_parity_ratio") == "#059669"

    def test_group_exposures_without_any_metric_are_not_declared_fair(self):
        svg = ranking_fairness_to_svg(results=[], group_metrics=GROUP_METRICS)

        assert ">FAIR<" not in svg

    def test_nothing_measured_says_so_instead_of_borrowing_either_verdict(self):
        """Could-not-check is its own state: not FAIR, and not UNFAIR either."""
        svg = ranking_fairness_to_svg(results=[], group_metrics=GROUP_METRICS)

        assert ">NOT ASSESSED<" in svg
        assert ">UNFAIR<" not in svg
        assert "no metric computed" in svg

    def test_nothing_measured_never_claims_the_metrics_passed(self):
        svg = ranking_fairness_to_svg(results=[], group_metrics=GROUP_METRICS)

        assert "All ranking fairness metrics pass" not in svg
        assert "No ranking fairness metric was supplied" in svg

    def test_the_not_assessed_headline_is_not_painted_green(self):
        """The colour carries the verdict as much as the word does.

        Read out of the rendered document rather than compared to a literal
        hex: the skin recolours every fill on the way out, so a hard-coded
        swatch would pin the skin rather than the verdict.
        """
        passing = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity", "value": 0.03, "threshold": 0.1}], GROUP_METRICS
        )
        green = re.search(r'fill="(#[0-9a-fA-F]{6})">FAIR<', passing)
        assert green, "the passing headline did not render a FAIR verdict to compare against"

        svg = ranking_fairness_to_svg(results=[], group_metrics=GROUP_METRICS)
        unknown = re.search(r'fill="(#[0-9a-fA-F]{6})">NOT ASSESSED<', svg)
        assert unknown, "the not-assessed headline did not render"

        assert unknown.group(1) != green.group(1)
        # And the state the adapter hands the template is the neutral slate,
        # not either verdict colour.
        assert _headline(0, 0)["color"] == "#64748b"


class TestGroupExposureTable:
    def test_group_exposures_and_positions_reach_the_page(self):
        svg = ranking_fairness_to_svg([], GROUP_METRICS)

        assert ">0.590<" in svg and ">0.320<" in svg
        assert "n=5" in svg
        assert ">2.0<" in svg and ">7.0<" in svg

    def test_a_wide_exposure_gap_earns_the_rerank_recommendation(self):
        wide = {
            "A": {"mean_exposure": 0.9, "count": 5},
            "B": {"mean_exposure": 0.2, "count": 5},
        }

        svg = ranking_fairness_to_svg([], wide)

        assert "Large exposure gap between groups" in svg

    def test_a_narrow_exposure_gap_does_not(self):
        narrow = {
            "A": {"mean_exposure": 0.55, "count": 5},
            "B": {"mean_exposure": 0.45, "count": 5},
        }

        svg = ranking_fairness_to_svg([], narrow)

        assert "Large exposure gap between groups" not in svg
        assert "exposure is well-distributed" in svg

    def test_a_zero_exposure_group_does_not_divide_by_zero(self):
        svg = ranking_fairness_to_svg(
            [], {"A": {"mean_exposure": 0.5, "count": 5}, "B": {"mean_exposure": 0.0, "count": 5}}
        )

        assert "<svg" in svg
        assert ">0.000<" in svg

    def test_only_the_first_eight_groups_are_tabulated(self):
        many = {f"g{i}": {"mean_exposure": 0.5, "count": 1} for i in range(10)}

        svg = ranking_fairness_to_svg([], many)

        assert ">g0<" in svg
        assert ">g9<" not in svg


class TestResultObjects:
    def test_a_result_object_is_read_through_to_dict(self):
        class RankingFairnessResult:
            def to_dict(self):
                return {
                    "metric_name": "exposure_parity",
                    "value": 0.4,
                    "threshold": 0.1,
                    "is_fair": False,
                }

        svg = ranking_fairness_to_svg([RankingFairnessResult()], GROUP_METRICS)

        assert ">UNFAIR<" in svg
        assert "EXPOSURE_PARITY" in svg

    def test_the_demo_path_renders_when_it_is_asked_for(self):
        # example=True is now required: a bare ranking_fairness_to_svg() used to
        # reach this fixture as a runtime fallback and rendered its invented
        # 2/3 pass rate as a real finding.
        svg = ranking_fairness_to_svg(example=True)

        assert "Ranking Fairness Report" in svg
        assert "<svg" in svg
