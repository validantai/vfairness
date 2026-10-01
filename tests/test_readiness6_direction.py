"""Readiness-6 pins: metric DIRECTION on the operational dashboard and the tracker.

One bug class, two more surfaces. A metric's better-direction was decided
locally, by assuming every fairness metric is a violation magnitude where lower
is better, and that assumption inverts the whole ratio family: the four-fifths
rule on ``disparate_impact_ratio`` / ``exposure_parity_ratio`` is a required
MINIMUM, so a value of 0.00 (the protected group is never selected) is the worst
possible reading and 1.00 is the best.

``evaluation.vfairness_metrics._metric_direction`` owns the precedence table and
is pinned by ``tests/test_metric_direction.py``. Nothing here re-derives a
direction; these tests assert that the two surfaces below ASK it, and that they
paint could-not-check as its own state rather than borrowing a verdict.

What was measured, 2026-09-10, before the fixes:

**dashboard.create_disparity_comparison / create_operational_view.** A store
holding ``demographic_parity`` 0.95 for gender_female and 0.02 for gender_male
rendered ``marker_color = ('#059669', '#dc2626')`` with a dashed reference line
at y=0.8 captioned "80% Rule". ``check_threshold("demographic_parity", 0.95,
0.8)`` answers FAIL and ``(..., 0.02, 0.8)`` answers PASS, so the near-total
parity gap was painted pass green and the near-perfect one fail red, under a
legend naming the EEOC four-fifths rule. ``create_operational_view()`` on the
same store rendered ``('#059669', '#f59e0b')``. Both reach the live Dash app,
the standalone HTML from ``to_html()``, and every embedded report figure.

**dashboard.create_intersectional_heatmap.** ``vals.append(val)`` sat outside
both loops, so ``vals`` held one element and ``zip(rows, cols, vals)`` truncated
to it: with four real intersectional values ingested the 2-D branch returned a
6x2 grid with 0 of 12 cells populated, all NaN, rendering an empty chart. The
row labels were wrong too, reading
``['F', 'F_race_black', 'F_race_white', 'M', 'M_race_black', 'M_race_white']``.

**tracker.detect_weekly_degradation.** ``higher_is_better`` defaulted to False
and neither library caller passed it. On a ``disparate_impact_ratio``:
Fridays at 0.10 against 0.30 rendered "DEGRADED / worst day mean: 0.300" (the
HEALTHIEST weekday); Fridays at PERFECT 1.00 against 0.85 rendered "DEGRADED /
worst day mean: 1.000"; Fridays at 0.10 against 0.15, a week in which every day
fails the four-fifths rule, rendered "STABLE / worst day mean: 0.150".

**tracker.get_explanation.** A four-fifths ratio falling 0.98 -> 0.40 got
severity "medium" and "Continue standard monitoring."; the same model recovering
0.40 -> 0.98 got severity "high" and "Investigate the upward trend".

Every class below carries BOTH a refusal test and an over-correction control. A
dashboard that paints everything slate, or a weekday check that never fires, is
as useless as the inversion, so the controls assert real colours and real
verdicts in both directions.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)
from vfairness.operations.monitoring.tracker import TemporalFairnessAnalyzer

go = pytest.importorskip("plotly.graph_objects", reason="dashboard needs the [dashboard] extra")

from vfairness.operations.reporting.dashboard import (  # noqa: E402
    DashboardConfig,
    FairnessDashboard,
)
from vfairness.operations.reporting.store import (  # noqa: E402
    MetricsStore,
    MetricsStoreConfig,
)

#: The dashboard's own palette, read off the config so a rename cannot make a
#: colour assertion silently vacuous.
_CFG = DashboardConfig()
PASS_GREEN = _CFG.color_pass
WARN_AMBER = _CFG.color_warn
FAIL_RED = _CFG.color_fail
UNKNOWN_SLATE = _CFG.color_unknown


def _store(metric: str, pairs) -> MetricsStore:
    """A store holding one reading per group for one metric.

    Privacy is off deliberately: with it on, every value is WITHHELD as
    ``unknown_size`` (no recorded group size) and the bars come back NaN, which
    would make a colour assertion prove nothing about direction.
    """
    now = datetime.now()
    df = pd.DataFrame(
        [{"timestamp": now, "metric": metric, "value": v, "group": g} for g, v in pairs]
    )
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(df, source="test", group_col="group")
    return store


def _bar_colors(fig) -> tuple:
    for trace in fig.data:
        if trace.type == "bar":
            return tuple(trace.marker.color)
    raise AssertionError("no bar trace was rendered")


def _captions(fig) -> list:
    return [a.text for a in fig.layout.annotations]


def _hlines(fig) -> list:
    return [float(s.y0) for s in fig.layout.shapes]


class TestDisparityBarDirection:
    """The bar colours and the reference line follow the metric's own direction."""

    def test_a_lower_is_better_gap_is_not_painted_green_at_0_95(self):
        # REFUSAL. The exact measured case: 0.95 is a near-total parity gap and
        # 0.02 is near-perfect parity, and the old ladder had them exactly
        # backwards.
        d = FairnessDashboard(
            _store("demographic_parity", [("gender_female", 0.95), ("gender_male", 0.02)])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("demographic_parity")

        assert _bar_colors(fig) == (FAIL_RED, PASS_GREEN)
        # ... and the verdict on the canvas agrees with the shared grader.
        assert check_threshold("demographic_parity", 0.95, 0.8)[0] is ThresholdOutcome.FAIL
        assert check_threshold("demographic_parity", 0.02, 0.8)[0] is ThresholdOutcome.PASS

    def test_the_four_fifths_caption_is_not_shown_over_a_gap_metric(self):
        # The caption was not merely the wrong colour, it was the wrong
        # STANDARD: it told the reader a selection-rate ratio rule had been
        # applied to a quantity that is not a ratio.
        d = FairnessDashboard(_store("demographic_parity", [("a", 0.95), ("b", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("demographic_parity")

        assert _hlines(fig) == [pytest.approx(_CFG.gap_pass_bound)]
        assert not any("80% Rule" in c for c in _captions(fig))
        assert any("Max gap" in c for c in _captions(fig))

    def test_a_ratio_metric_still_gets_the_four_fifths_ladder(self):
        # OVER-CORRECTION CONTROL. The family the original ladder was written
        # for must keep working, in all three bands and with its real caption.
        d = FairnessDashboard(
            _store("disparate_impact_ratio", [("a", 0.95), ("b", 0.70), ("c", 0.02)])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("disparate_impact_ratio")

        assert _bar_colors(fig) == (PASS_GREEN, WARN_AMBER, FAIL_RED)
        assert _hlines(fig) == [pytest.approx(_CFG.ratio_pass_bound)]
        assert any("80% Rule" in c for c in _captions(fig))

    def test_a_gap_metric_still_grades_in_all_three_bands(self):
        # OVER-CORRECTION CONTROL, the mirror. Painting everything slate would
        # pass the refusal test above and be just as useless.
        d = FairnessDashboard(_store("demographic_parity", [("a", 0.02), ("b", 0.15), ("c", 0.95)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("demographic_parity")

        assert _bar_colors(fig) == (PASS_GREEN, WARN_AMBER, FAIL_RED)

    def test_an_unresolvable_direction_is_slate_and_says_so(self):
        # Three states, never two. No bound is drawn, because drawing one
        # asserts a comparison nobody made.
        d = FairnessDashboard(_store("mystery_metric", [("a", 0.95), ("b", 0.70), ("c", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("mystery_metric")

        assert _bar_colors(fig) == (UNKNOWN_SLATE, UNKNOWN_SLATE, UNKNOWN_SLATE)
        assert _hlines(fig) == []
        assert any("NOT GRADED" in c for c in _captions(fig))

    def test_an_unmeasurable_value_is_slate_not_the_failure_red(self):
        # ``nan >= 0.8`` is False, so an unmeasured value fell through to the
        # fail red, which is a verdict about a number nobody has.
        d = FairnessDashboard(_store("demographic_parity", [("a", float("nan")), ("b", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_disparity_comparison("demographic_parity")

        assert _bar_colors(fig) == (UNKNOWN_SLATE, PASS_GREEN)


class TestOperationalViewDirection:
    """The Tier-2 composite picks its metric up off the store, so it cannot assume a family."""

    def test_the_group_disparity_tile_is_not_inverted(self):
        d = FairnessDashboard(
            _store("demographic_parity", [("gender_female", 0.95), ("gender_male", 0.02)])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_operational_view()

        assert _bar_colors(fig) == (FAIL_RED, PASS_GREEN)

    def test_the_ratio_family_still_reads_green_at_0_95(self):
        # OVER-CORRECTION CONTROL.
        d = FairnessDashboard(_store("disparate_impact_ratio", [("a", 0.95), ("b", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = d.create_operational_view()

        assert _bar_colors(fig) == (PASS_GREEN, FAIL_RED)

    def test_the_exported_html_does_not_claim_the_four_fifths_rule_over_a_gap(self):
        # The artifact that leaves the building. ``to_html`` is what gets mailed
        # around and embedded in reports, and it carried the "80% Rule" caption
        # over a demographic-parity gap.
        d = FairnessDashboard(_store("demographic_parity", [("a", 0.95), ("b", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            html = d.to_html(tier="operational")

        assert "80% Rule" not in html
        assert "Max gap" in html

    def test_the_exported_html_still_shows_the_rule_for_a_real_ratio(self):
        # OVER-CORRECTION CONTROL on the same artifact.
        d = FairnessDashboard(_store("disparate_impact_ratio", [("a", 0.95), ("b", 0.02)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            html = d.to_html(tier="operational")

        assert "80% Rule" in html


class TestIntersectionalHeatmapCells:
    """A chart that renders nothing is not a chart that found nothing."""

    ROWS = [
        ("gender_F", 0.10),
        ("gender_M", 0.20),
        ("race_black", 0.30),
        ("race_white", 0.05),
        ("gender_F_race_black", 0.40),
        ("gender_F_race_white", 0.11),
        ("gender_M_race_black", 0.35),
        ("gender_M_race_white", 0.06),
    ]

    def _heatmap(self, metric="demographic_parity"):
        d = FairnessDashboard(_store(metric, self.ROWS))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return d.create_intersectional_heatmap(metric, "gender", "race").data[0]

    def test_every_ingested_intersection_reaches_a_cell(self):
        # REFUSAL. Measured before the fix: a 6x2 grid, 0 of 12 cells populated.
        hm = self._heatmap()
        z = np.asarray(hm.z, dtype=float)

        assert z.shape == (2, 2)
        assert int(np.count_nonzero(~np.isnan(z))) == 4

    def test_the_cells_hold_the_values_that_were_ingested(self):
        # Not just "populated": the RIGHT numbers in the RIGHT cells.
        hm = self._heatmap()
        z = np.asarray(hm.z, dtype=float)
        rows = list(hm.y)
        cols = list(hm.x)

        expected = {
            ("F", "black"): 0.40,
            ("F", "white"): 0.11,
            ("M", "black"): 0.35,
            ("M", "white"): 0.06,
        }
        for (r, c), v in expected.items():
            assert z[rows.index(r), cols.index(c)] == pytest.approx(v)

    def test_the_axis_labels_are_levels_not_intersections(self):
        # ``g.startswith(attribute_1)`` also matched every composite name, so
        # the intersections were listed as if they were levels of attribute_1.
        hm = self._heatmap()

        assert list(hm.y) == ["F", "M"]
        assert list(hm.x) == ["black", "white"]

    def test_a_lower_is_better_heatmap_does_not_paint_the_largest_gap_green(self):
        # ``colorscale="RdYlGn"`` maps the MINIMUM to dark red and the MAXIMUM
        # to dark green, so a zero gap was the darkest red cell on the grid.
        hm = self._heatmap("demographic_parity")

        assert hm.reversescale is True

    def test_a_ratio_heatmap_keeps_the_unreversed_ramp(self):
        # OVER-CORRECTION CONTROL: on a four-fifths ratio, high IS green.
        hm = self._heatmap("disparate_impact_ratio")

        assert hm.reversescale is False

    def test_an_unresolvable_direction_gets_a_ramp_that_claims_nothing(self):
        hm = self._heatmap("mystery_metric")
        endpoints = {hm.colorscale[0][1], hm.colorscale[-1][1]}

        # A single-hue magnitude ramp, with no green end to read as "good".
        assert endpoints == {"rgb(255,255,255)", "rgb(0,0,0)"}


def _weekly(friday: float, other: float, metric: str, days: int = 28):
    """A 4-week analyzer whose Fridays differ from the rest of the week."""
    analyzer = TemporalFairnessAnalyzer(lookback_days=90)
    for i in range(days):
        # 2025-01-06 is a Monday, so dayofweek 4 is Friday.
        date = pd.Timestamp("2025-01-06") + pd.Timedelta(days=i)
        analyzer.update_daily_metrics(date, {metric: friday if date.dayofweek == 4 else other})
    return analyzer


def _svg_texts(analyzer, metric):
    from vfairness.rendering.adapters_monitoring import temporal_analysis_to_svg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = temporal_analysis_to_svg(analyzer, metric_names=[metric])
    return [m.group(1).strip() for m in re.finditer(r">([^<>]{1,140})<", svg)]


class TestWeeklyDegradationDirection:
    """``worst day`` must mean the worst day for THIS metric."""

    RATIO = "disparate_impact_ratio"

    def test_the_worst_day_is_the_collapsed_day_not_the_healthy_one(self):
        # REFUSAL. Measured before the fix: (True, 0.30), naming the best
        # weekday of the week as the worst.
        analyzer = _weekly(0.10, 0.30, self.RATIO)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degraded, worst = analyzer.detect_weekly_degradation(self.RATIO)

        assert degraded is True
        assert worst == pytest.approx(0.10)

    def test_a_perfect_weekday_is_not_reported_as_degradation(self):
        # REFUSAL, the false-alarm polarity. Measured before the fix:
        # (True, 1.00), i.e. perfect parity flagged as the worst day.
        analyzer = _weekly(1.00, 0.85, self.RATIO)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degraded, worst = analyzer.detect_weekly_degradation(self.RATIO)

        assert degraded is False
        assert worst == pytest.approx(0.85)

    def test_a_catastrophic_friday_is_not_certified_stable(self):
        # REFUSAL, the false-PASS polarity. Measured before the fix:
        # (False, 0.15), on a week where every single day fails four-fifths.
        analyzer = _weekly(0.10, 0.15, self.RATIO)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degraded, worst = analyzer.detect_weekly_degradation(self.RATIO)

        assert degraded is True
        assert worst == pytest.approx(0.10)

    @pytest.mark.parametrize(
        ("friday", "other", "expect_degraded", "expect_worst"),
        [(0.10, 0.30, True, 0.10), (1.00, 0.85, False, 0.85), (0.10, 0.15, True, 0.10)],
    )
    def test_the_rendered_chart_agrees(self, friday, other, expect_degraded, expect_worst):
        # The SVG is the artifact an auditor reads, and all three cases were
        # visible there: "DEGRADED / worst day mean: 0.300" and so on.
        texts = _svg_texts(_weekly(friday, other, self.RATIO), self.RATIO)

        assert ("DEGRADED" in texts) is expect_degraded
        assert ("STABLE" in texts) is (not expect_degraded)
        assert f"worst day mean: {expect_worst:.3f}" in texts

    def test_a_lower_is_better_metric_still_fires_on_its_bad_day(self):
        # OVER-CORRECTION CONTROL. A check that never fires is as useless as an
        # inverted one.
        analyzer = _weekly(0.40, 0.05, "demographic_parity")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degraded, worst = analyzer.detect_weekly_degradation("demographic_parity")

        assert degraded is True
        assert worst == pytest.approx(0.40)

    def test_a_flat_healthy_ratio_is_not_flagged(self):
        # OVER-CORRECTION CONTROL, the other polarity.
        analyzer = _weekly(0.95, 0.95, self.RATIO)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            degraded, worst = analyzer.detect_weekly_degradation(self.RATIO)

        assert degraded is False
        assert worst == pytest.approx(0.95)

    def test_an_unresolvable_direction_could_not_check_and_warns(self):
        analyzer = _weekly(0.10, 0.30, "mystery_metric")

        with pytest.warns(UserWarning, match="no known better-direction"):
            degraded, worst = analyzer.detect_weekly_degradation("mystery_metric")

        assert degraded is None
        assert np.isnan(worst)

    def test_an_explicit_override_is_still_honoured(self):
        # A caller who knows something the name does not keeps their answer.
        analyzer = _weekly(0.10, 0.30, "mystery_metric")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hib = analyzer.detect_weekly_degradation("mystery_metric", higher_is_better=True)
            lib = analyzer.detect_weekly_degradation("mystery_metric", higher_is_better=False)

        assert hib == (True, pytest.approx(0.10))
        assert lib == (True, pytest.approx(0.30))


def _trend(values, metric):
    analyzer = TemporalFairnessAnalyzer(lookback_days=90)
    for i, v in enumerate(values):
        analyzer.update_daily_metrics(
            pd.Timestamp("2025-01-06") + pd.Timedelta(days=i), {metric: v}
        )
    return analyzer


_N = 30
_FALLING = [0.98 - 0.58 * i / (_N - 1) for i in range(_N)]
_RISING = [0.40 + 0.58 * i / (_N - 1) for i in range(_N)]
_GAP_RISING = [0.01 + 0.30 * i / (_N - 1) for i in range(_N)]
_GAP_FALLING = [0.31 - 0.30 * i / (_N - 1) for i in range(_N)]


class TestTemporalExplanationDirection:
    """A collapsing four-fifths ratio is not "continue standard monitoring"."""

    RATIO = "disparate_impact_ratio"

    def _report(self, values, metric):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _trend(values, metric).get_explanation(metric)

    def test_a_collapsing_ratio_is_escalated(self):
        # REFUSAL. Measured before the fix: severity "medium" and "Continue
        # standard monitoring." for 0.98 -> 0.40.
        report = self._report(_FALLING, self.RATIO)

        assert report.severity == "high"
        assert "Investigate the decreasing trend" in report.explanations[0].recommendation

    def test_a_recovering_ratio_is_not_escalated(self):
        # REFUSAL, the mirror. Measured before the fix: severity "high" and
        # "Investigate the upward trend" for the same model getting BETTER.
        report = self._report(_RISING, self.RATIO)

        assert report.severity != "high"
        assert report.explanations[0].recommendation == "Continue standard monitoring."

    def test_a_widening_gap_is_still_escalated(self):
        # OVER-CORRECTION CONTROL. The lower-is-better family the original rule
        # was written for must keep escalating.
        report = self._report(_GAP_RISING, "demographic_parity")

        assert report.severity == "high"
        assert "Investigate the increasing trend" in report.explanations[0].recommendation

    def test_a_closing_gap_is_not_escalated(self):
        report = self._report(_GAP_FALLING, "demographic_parity")

        assert report.severity != "high"
        assert report.explanations[0].recommendation == "Continue standard monitoring."

    def test_an_unresolvable_direction_grades_neither_way(self):
        report = self._report(_GAP_RISING, "mystery_metric")

        assert report.severity == "info"
        guide = report.explanations[0].interpretation_guide
        assert "NOT GRADED" in guide
        assert "not graded here" in report.explanations[0].recommendation

    def test_the_guide_names_the_direction_that_is_worse_for_this_metric(self):
        # The prose said "an increasing trend in disparity metrics signals
        # gradual fairness degradation" for EVERY metric, including the ratios
        # where increasing is recovery.
        ratio_guide = self._report(_FALLING, self.RATIO).explanations[0].interpretation_guide
        gap_guide = (
            self._report(_GAP_RISING, "demographic_parity").explanations[0].interpretation_guide
        )

        assert "decreasing signals gradual" in ratio_guide
        assert "increasing signals gradual" in gap_guide


class TestExplanationCoverageIsDeclared:
    """One report covers ONE metric, and it has to say so.

    ``get_explanation`` defaulted its metric name to "demographic_parity", so an
    analyzer holding several metrics was silently narrowed to one of them.
    Measured 2026-09-10, 20 days, two metrics with OPPOSITE directions
    (``demographic_parity`` flat at 0.02, clean; ``disparate_impact`` falling
    0.95 -> 0.19, severe): the bare call returned "demographic_parity over 20
    days: mean=0.0200, trend=stable." at severity ``info`` with ZERO warnings,
    and never named ``disparate_impact``, which the same method grades ``high``.
    """

    @staticmethod
    def _two_metric_analyzer(days: int = 20):
        analyzer = TemporalFairnessAnalyzer(lookback_days=90)
        for i in range(days):
            date = pd.Timestamp("2025-01-06") + pd.Timedelta(days=i)
            analyzer.update_daily_metrics(
                date,
                {
                    "demographic_parity": 0.02,
                    "disparate_impact": 0.95 - (0.76 * i / (days - 1)),
                },
            )
        return analyzer

    def test_the_metric_name_cannot_be_left_to_a_default(self):
        analyzer = self._two_metric_analyzer()

        with pytest.raises(TypeError, match="metric_name"):
            analyzer.get_explanation()

    def test_a_partial_report_names_the_metrics_it_did_not_assess(self):
        analyzer = self._two_metric_analyzer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.get_explanation("demographic_parity")

        assert "Covers 1 of 2 tracked metric(s)" in report.summary
        assert "disparate_impact" in report.summary
        assert any("disparate_impact" in r for r in report.recommendations)

    def test_the_metric_it_skipped_is_the_severe_one(self):
        # The finding the silent default hid, still findable when asked for.
        analyzer = self._two_metric_analyzer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = analyzer.get_explanation("demographic_parity")
            severe = analyzer.get_explanation("disparate_impact")

        assert clean.severity == "info"
        assert severe.severity == "high"

    def test_a_single_metric_analyzer_makes_no_coverage_noise(self):
        # OVER-CORRECTION CONTROL. A disclosure printed when there is nothing to
        # disclose trains the reader to skip it.
        analyzer = TemporalFairnessAnalyzer(lookback_days=90)
        for i in range(20):
            analyzer.update_daily_metrics(
                pd.Timestamp("2025-01-06") + pd.Timedelta(days=i), {"demographic_parity": 0.02}
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.get_explanation("demographic_parity")

        assert "Covers 1 of" not in report.summary
        assert not any("get_tracked_metrics" in r for r in report.recommendations)

    def test_get_tracked_metrics_lists_the_columns_and_not_the_date(self):
        assert self._two_metric_analyzer().get_tracked_metrics() == [
            "demographic_parity",
            "disparate_impact",
        ]
        assert TemporalFairnessAnalyzer().get_tracked_metrics() == []
