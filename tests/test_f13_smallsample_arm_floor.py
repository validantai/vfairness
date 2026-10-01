"""F13, 2026-09-30. One defect family on three surfaces, plus two routing flags.

A FLOOR COUNTING ROWS WHERE THE QUANTITY NEEDS LABELS.

``FairnessMonitor``'s only floor was ``min_samples``, a WHOLE-GROUP floor applied
with ``if len(grp) < self.config.min_samples: continue``. Neither label-dependent
metric is a rate over the group: a true-positive rate is divided by the group's
POSITIVE-LABEL rows and a false-positive rate by its NEGATIVE-LABEL rows, and
those are labelled subsets that can hold one row inside a group of forty. So the
guard counted one thing and the arithmetic divided by another.

Measured on the tree before the fix, ``min_samples=30``, group A 40 rows of which
EXACTLY ONE carries label = 1 and it is selected, group B 20 positive-label rows
all selected plus 20 negative-label rows none of which is::

    group A: n=40  pos-arm n=1   TPR 1.0000  neg-arm n=39  FPR 0.0000
    group B: n=40  pos-arm n=20  TPR 1.0000  neg-arm n=20  FPR 0.0000
    compute_equal_opportunity -> 0.0   warnings []
    compute_equalized_odds    -> 0.0   warnings []
    update_and_check          -> both 0.0, both alerts False, any_alert False,
                                 excluded_groups {}

Two readings of PERFECT equality, two guardrails recorded as applied and passed,
and one side of both is a single observation.

The third surface is the same family at the RENDER boundary. The label-conditional
denominators ``operations/cicd/gate.py`` records live on the LEVEL decisions, and
``hierarchical_gate_to_svg`` read only the hierarchy's own row-count list, so a
true-positive rate certified over ONE row drew no small-sample panel at all; and
``small_sample_check_ran``'s three states rendered byte-identical canvases.

Every refusal below is paired with an OVER-CORRECTION CONTROL asserting a real
number, because reusing ``min_samples`` as the arm floor reddens three of this
repo's own controls and that would have been the wrong guard.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.gate import (
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    IntersectionalGateDecision,
    ModelFairnessGate,
    SmallSampleWarning,
)
from vfairness.operations.monitoring.tracker import (
    _MIN_ARM_ROWS,
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
)
from vfairness.rendering.adapters_workflow import hierarchical_gate_to_svg


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _monitor(**cfg) -> FairnessMonitor:
    cfg.setdefault("min_samples", 30)
    cfg.setdefault("metrics_to_track", ["equalized_odds", "equal_opportunity"])
    return FairnessMonitor(config=FairnessMonitorConfig(**cfg))


# ===========================================================================
# Fixtures. Deterministic, so every number asserted below is exact.
# ===========================================================================


def _one_observation_arm() -> pd.DataFrame:
    """Group A: 40 rows, ONE label=1 row, selected. Group B: 20 pos all selected.

    Both groups clear min_samples=30 on ROW count. Both TPRs are 1.0 and both
    FPRs are 0.0, so both metrics come out 0.0, and A's TPR was divided by 1.
    """
    rows = [
        {"group_gender": "A", "label": 1 if i == 0 else 0, "prediction": 1 if i == 0 else 0}
        for i in range(40)
    ]
    rows += [{"group_gender": "B", "label": 1, "prediction": 1} for _ in range(20)]
    rows += [{"group_gender": "B", "label": 0, "prediction": 0} for _ in range(20)]
    return pd.DataFrame(rows)


def _twenty_row_arms() -> pd.DataFrame:
    """40 rows per group, 20 per arm, a REAL true-positive-rate gap of 0.6.

    The same shape as tests/test_bgl3_operations_2.py::_both_groups_measurable,
    which is one of the three controls that reusing min_samples would redden.
    """
    rows = [
        {"prediction": int(i < 32), "label": int(i % 2 == 0), "group_gender": "M"}
        for i in range(40)
    ]
    rows += [
        {"prediction": int(i < 8), "label": int(i % 2 == 0), "group_gender": "F"} for i in range(40)
    ]
    return pd.DataFrame(rows)


def _arms_of_exactly(n_pos: int) -> pd.DataFrame:
    """Two 40-row groups whose positive arms hold *n_pos* rows each.

    Group M selects every positive row (TPR 1.0), group F selects none (TPR 0.0),
    so the equal-opportunity gap is exactly 1.0 whenever it is measurable.
    """
    rows = []
    for grp, select in (("M", 1), ("F", 0)):
        rows += [{"prediction": select, "label": 1, "group_gender": grp} for _ in range(n_pos)]
        rows += [{"prediction": 0, "label": 0, "group_gender": grp} for _ in range(40 - n_pos)]
    return pd.DataFrame(rows)


# ===========================================================================
# Surface 1: compute_equal_opportunity
# ===========================================================================


class TestEqualOpportunityRefusesAnArmOfOne:
    def test_a_true_positive_rate_divided_by_one_is_not_perfect_parity(self):
        """REFUSAL PIN. 0.0 with zero warnings before; nan with the count now."""
        value, warned = _caught(
            lambda: _monitor().compute_equal_opportunity(_one_observation_arm(), "group_gender")
        )

        assert math.isnan(value), "a TPR over one row was published as perfect parity"
        assert len(warned) == 1, warned
        assert "1 row(s)" in warned[0], "the arm's real count is not named"
        assert "positive-label rows" in warned[0], "the arm is not named for what it counts"
        assert "min_samples" in warned[0], "the warning does not say why the group floor passed"
        assert "0.0" in warned[0], "the warning does not say what 0.0 would have read as"

    def test_the_refusal_reaches_the_snapshot_as_could_not_check(self):
        """The value alone is not the fix: the guardrail must not read as applied."""
        snap, warned = _caught(lambda: _monitor().update_and_check(_one_observation_arm()))

        assert math.isnan(snap.metrics["equal_opportunity_group_gender"])
        assert math.isnan(snap.metrics["equalized_odds_group_gender"])
        # An absent alerts entry is this file's could-not-check state. A False
        # here would be the record of a guardrail that ran and passed.
        assert snap.alerts == {}
        assert snap.any_alert is None, "a window that compared nothing reported a clean bill"
        assert snap.uncompared_metrics == [
            "equalized_odds_group_gender",
            "equal_opportunity_group_gender",
        ]
        assert len(warned) == 2, warned

    def test_control_twenty_row_arms_still_measure_their_exact_gap(self):
        """OVER-CORRECTION CONTROL. M TPR 1.0, F TPR 0.4, gap exactly 0.6, silent.

        This is the shape reusing min_samples=30 as the arm floor turns red.
        """
        value, warned = _caught(
            lambda: _monitor().compute_equal_opportunity(_twenty_row_arms(), "group_gender")
        )

        assert value == pytest.approx(0.6, abs=1e-9)
        assert warned == []

    def test_control_an_arm_of_exactly_the_floor_is_measured(self):
        """OVER-CORRECTION CONTROL. The floor is inclusive, and the gap is 1.0.

        At _MIN_ARM_ROWS the rate is measured; one row below it is refused. The
        same frame either side of the bound, so only the floor decides.
        """
        at_floor, clean = _caught(
            lambda: _monitor().compute_equal_opportunity(
                _arms_of_exactly(_MIN_ARM_ROWS), "group_gender"
            )
        )
        below, warned = _caught(
            lambda: _monitor().compute_equal_opportunity(
                _arms_of_exactly(_MIN_ARM_ROWS - 1), "group_gender"
            )
        )

        assert at_floor == pytest.approx(1.0, abs=1e-9), "the floor refused its own boundary"
        assert clean == []
        assert math.isnan(below)
        assert f"{_MIN_ARM_ROWS - 1} row(s)" in warned[0]

    def test_control_a_zero_row_arm_keeps_the_older_refusal_and_its_single_warning(self):
        """A quantity that does not EXIST is not a quantity measured over too few.

        tests/test_bgl3_operations_2.py asserts exactly one warning on this frame,
        and the len(valid) < 2 refusal owns it. The new guard uses 0 < n on
        purpose so it cannot put a second wording on the same number.
        """
        frame = pd.DataFrame(
            [{"prediction": int(i % 2 == 0), "label": 1, "group_gender": "M"} for i in range(40)]
            + [{"prediction": int(i % 4 == 0), "label": 0, "group_gender": "F"} for i in range(40)]
        )
        value, warned = _caught(lambda: _monitor().compute_equal_opportunity(frame, "group_gender"))

        assert math.isnan(value)
        assert len(warned) == 1, warned
        assert "no positive label in this window" in warned[0]
        assert "row(s) in the denominator" not in warned[0]


# ===========================================================================
# Surface 2: compute_equalized_odds, the sibling door
# ===========================================================================


class TestEqualizedOddsRefusesTheSameArm:
    def test_the_max_of_two_arms_is_refused_above_the_dispatch(self):
        """REFUSAL PIN. Either arm of the max() can be the underpowered one."""
        value, warned = _caught(
            lambda: _monitor().compute_equalized_odds(_one_observation_arm(), "group_gender")
        )

        assert math.isnan(value)
        assert len(warned) == 1, warned
        assert "1 row(s)" in warned[0]
        assert "positive-label rows" in warned[0]

    def test_a_thin_negative_arm_is_refused_too_not_only_the_positive_one(self):
        """The FPR leg is the sibling door, and it loses the max just as quietly.

        Both groups hold 20 positive-label rows, so the TPR arm is healthy; group
        F holds THREE negative-label rows. Equalized odds is the max of the two
        arms, so a coarse FPR arm can decide the published number outright.
        """
        rows = [{"prediction": int(i < 10), "label": 1, "group_gender": "M"} for i in range(20)]
        rows += [{"prediction": 0, "label": 0, "group_gender": "M"} for _ in range(20)]
        rows += [{"prediction": int(i < 10), "label": 1, "group_gender": "F"} for i in range(20)]
        rows += [{"prediction": 0, "label": 0, "group_gender": "F"} for _ in range(3)]
        # F holds 23 rows, which still clears min_samples=20 below.
        value, warned = _caught(
            lambda: _monitor(min_samples=20).compute_equalized_odds(
                pd.DataFrame(rows), "group_gender"
            )
        )

        assert math.isnan(value), "an FPR over three rows decided the published max"
        assert len(warned) == 1, warned
        assert "3 row(s)" in warned[0]
        assert "negative-label rows" in warned[0], "the FPR leg is not named"

    def test_control_twenty_row_arms_still_measure_their_exact_gap(self):
        """OVER-CORRECTION CONTROL. The same 0.6, and silence."""
        value, warned = _caught(
            lambda: _monitor().compute_equalized_odds(_twenty_row_arms(), "group_gender")
        )

        assert value == pytest.approx(0.6, abs=1e-9)
        assert warned == []

    def test_control_the_readiness_fixture_keeps_its_exact_one_point_zero(self):
        """OVER-CORRECTION CONTROL, and the third of the three named ones.

        600 rows in each of two groups selected 1-in-2, plus 46 rows in 'native'
        never selected, whose arms hold 23 rows each. Every metric is a real
        number, the gap is exactly 1.0 on both arms, and NOTHING warns. This is
        the control that reusing min_samples=30 as the arm floor turns red.
        """
        rows = []
        for grp in ("white", "black"):
            rows.append(
                pd.DataFrame({"prediction": [1, 0] * 300, "label": [1, 0] * 300, "group_race": grp})
            )
        rows.append(
            pd.DataFrame({"prediction": [0] * 46, "label": [1, 0] * 23, "group_race": "native"})
        )
        frame = pd.concat(rows, ignore_index=True)

        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                window_size=5000,
                min_samples=30,
                metrics_to_track=["equalized_odds", "equal_opportunity"],
            )
        )
        snap, warned = _caught(lambda: monitor.update_and_check(frame))

        assert snap.metrics["equalized_odds_group_race"] == pytest.approx(1.0)
        assert snap.metrics["equal_opportunity_group_race"] == pytest.approx(1.0)
        assert snap.alerts == {
            "equalized_odds_group_race": True,
            "equal_opportunity_group_race": True,
        }
        assert snap.any_alert is True
        assert warned == []


# ===========================================================================
# Surface 3: hierarchical_gate_to_svg
# ===========================================================================


def _clean_hierarchy():
    y_true = np.array([1, 0] * 100)
    gate = ModelFairnessGate(config=GateConfig(min_group_size=30))
    hcfg = HierarchicalGateConfig(
        min_group_size=30, check_single_attributes=True, check_intersections=False
    )
    return gate.evaluate_hierarchical(
        y_true,
        y_true.copy(),
        {"gender": np.array(["m"] * 100 + ["f"] * 100)},
        hierarchical_config=hcfg,
    )


def _strip_ts(svg: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "TS", svg)


class TestTheGateCanvasCarriesTheSmallSampleState:
    def test_never_run_does_not_render_as_checked_and_clean(self):
        """REFUSAL PIN. The three states rendered byte-identical canvases.

        Measured before the fix with the level results and the empty warning list
        held FIXED and only the field varied: True, False and None all produced
        sha256 c47334584ab93abd, len 19073.
        """
        import copy

        base = _clean_hierarchy()
        rendered = {}
        for state in (True, False, None):
            d = copy.deepcopy(base)
            d.small_sample_check_ran = state
            d.small_sample_unsized = []
            rendered[state] = _strip_ts(hierarchical_gate_to_svg(d))

        assert len(set(rendered.values())) == 3, "the three states still collapse"
        assert "SMALL-SAMPLE CHECK: RAN" in rendered[True]
        assert "SMALL-SAMPLE CHECK: NOT RUN" in rendered[False]
        assert "SMALL-SAMPLE CHECK: NOT RECORDED" in rendered[None]
        assert "NOT CHECKED, not" in rendered[False]
        assert "Could not check" in rendered[None]
        # The clean state says so positively, so its silence cannot be mistaken
        # for the other two.
        assert "nothing fell below it" in rendered[True]

    def test_a_minimum_nothing_can_fall_below_is_named_on_the_canvas(self):
        import copy

        d = copy.deepcopy(_clean_hierarchy())
        d.small_sample_check_ran = False
        d.small_sample_vacuous_minimum = 1
        svg = hierarchical_gate_to_svg(d)

        assert "SMALL-SAMPLE CHECK: NOT RUN" in svg
        assert "The configured minimum is 1" in svg
        assert "could not have failed" in svg

    def test_what_a_ran_does_not_cover_is_named(self):
        import copy

        d = copy.deepcopy(_clean_hierarchy())
        d.small_sample_check_ran = True
        d.small_sample_unsized = ["the four intersection cells of gender x race"]
        svg = hierarchical_gate_to_svg(d)

        assert "Never sized:" in svg
        assert "intersection cells of gender x race" in svg

    def test_a_rate_denominator_of_one_row_reaches_the_canvas(self):
        """REFUSAL PIN, the label-conditional half.

        Measured before the fix on two groups of 100 rows with group f holding ONE
        positive-label row, min_group_size=30, equalized_odds_difference gated: the
        level decisions each carried the leg warning and blocked on it, while
        'true-positive rate leg' and 'SMALL-SAMPLE' were both absent from the SVG.
        """
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=["equalized_odds_difference", "demographic_parity_difference"],
                thresholds={
                    "equalized_odds_difference": 0.1,
                    "demographic_parity_difference": 0.1,
                },
                min_group_size=30,
            )
        )
        y_true = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] + [0] * 99)])
        decision = gate.evaluate_hierarchical(
            y_true,
            y_true.copy(),
            {"gender": np.array(["m"] * 100 + ["f"] * 100)},
            hierarchical_config=HierarchicalGateConfig(
                min_group_size=30, check_single_attributes=True, check_intersections=False
            ),
        )
        # The hierarchy's OWN list is empty: the legs live on the level decisions.
        assert decision.small_sample_warnings == []
        assert any(lev.small_sample_warnings for lev in decision.level_results.values()), (
            "the gate no longer records the leg at all; this pin has lost its subject"
        )

        svg = hierarchical_gate_to_svg(decision)

        assert "true-positive rate leg" in svg
        assert "1 row(s) in the denominator (min 30)" in svg
        assert "for equalized_odds_difference" in svg, "the leg is not tied to its metric"
        # Printed once, not once per level that evaluated the metric.
        assert svg.count("1 row(s) in the denominator (min 30)") == 1

    def test_the_heading_no_longer_invents_a_floor_the_decision_never_applied(self):
        """Measured before the fix: the heading read "(30)" directly above a row
        whose own minimum is 200."""
        d = IntersectionalGateDecision(
            approved=True,
            status=GateStatus.APPROVED,
            level_results={},
            small_sample_warnings=[
                SmallSampleWarning(group_name="tiny", sample_size=3, minimum_recommended=200)
            ],
            hierarchical_config=None,
        )
        svg = hierarchical_gate_to_svg(d)

        assert "(30)" not in svg
        assert "not recorded here" in svg
        assert "tiny: 3 row(s) in the denominator (min 200)" in svg

    def test_control_a_configured_floor_is_still_printed_as_the_floor(self):
        """OVER-CORRECTION CONTROL. A known minimum is stated, not withheld."""
        svg = hierarchical_gate_to_svg(_clean_hierarchy())

        assert "the minimum of 30" in svg
        assert "not recorded here" not in svg

    def test_control_a_gate_handed_nothing_draws_no_small_sample_panel(self):
        """OVER-CORRECTION CONTROL. No decision means no small-sample claim.

        The not_assessable band already says no level was evaluated; a NOT RUN
        panel here would imply a gate ran and skipped one check.
        """
        svg = hierarchical_gate_to_svg(None)

        assert "SMALL-SAMPLE CHECK" not in svg
        assert "no level" in svg


# ===========================================================================
# Flag 1: the lookback window
# ===========================================================================


class TestTheLookbackCannotEatTheRowItWasGiven:
    def test_a_lookback_of_zero_is_refused_before_any_reading_is_lost(self):
        """REFUSAL PIN. Measured before the fix: one update at lookback_days=0
        left 0 rows in the window with ZERO warnings, and all four consumers then
        reported could-not-check for a history supplied in full."""
        with pytest.raises(ValueError, match="at least 1 day"):
            TemporalFairnessAnalyzer(lookback_days=0)
        with pytest.raises(ValueError, match="at least 1 day"):
            TemporalFairnessAnalyzer(lookback_days=-5)

    def test_control_a_lookback_of_one_keeps_the_row_it_records(self):
        """OVER-CORRECTION CONTROL. 1 is the smallest window that works, so the
        bound is exactly where it belongs."""
        analyzer = TemporalFairnessAnalyzer(lookback_days=1)
        analyzer.update_daily_metrics(pd.Timestamp("2025-03-01"), {"demographic_parity": 0.9})

        assert len(analyzer.to_dataframe()) == 1
        assert analyzer.get_metric_summary("demographic_parity")["n_days"] == 1

    def test_control_the_default_analyzer_still_measures_a_real_trend(self):
        """OVER-CORRECTION CONTROL. 14 rising days, slope exactly 0.01/day."""
        analyzer = TemporalFairnessAnalyzer(lookback_days=30)
        for i in range(14):
            analyzer.update_daily_metrics(
                pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                {"demographic_parity": 0.01 + i * 0.01},
            )

        direction, slope = _caught(lambda: analyzer.detect_trend("demographic_parity"))[0]
        assert direction == "increasing"
        assert slope == pytest.approx(0.01, abs=1e-12)

    @pytest.mark.parametrize("n_rows", [0, 1])
    def test_an_emptied_or_single_point_window_is_could_not_check_on_all_four(self, n_rows):
        """The four consumers named in the routing flag, on a window the lookback
        pruned down to nothing and to one point. None of them answers with a PASS
        or a clean trend."""
        analyzer = TemporalFairnessAnalyzer(lookback_days=30)
        for i in range(60):
            analyzer.update_daily_metrics(
                pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                {"demographic_parity": 0.01 + i * 0.01},
            )
        # One row 180 days on prunes the entire history behind it.
        analyzer.update_daily_metrics(pd.Timestamp("2025-09-01"), {"demographic_parity": 0.61})
        if n_rows == 0:
            analyzer._daily_metrics = analyzer._daily_metrics.iloc[0:0]
        assert len(analyzer.to_dataframe()) == n_rows

        (direction, slope), trend_warned = _caught(
            lambda: analyzer.detect_trend("demographic_parity")
        )
        assert direction == "not_assessed" and math.isnan(slope)
        assert trend_warned

        (degraded, worst), week_warned = _caught(
            lambda: analyzer.detect_weekly_degradation("demographic_parity")
        )
        assert degraded is None, "an unmeasurable week reported a weekday verdict"
        assert week_warned

        forecast, fc_warned = _caught(
            lambda: analyzer.forecast_metric("demographic_parity", days_ahead=3)
        )
        assert len(forecast) == 3
        assert all(math.isnan(v) for _, v in forecast), "a forecast was anchored on nothing"
        assert fc_warned

        pattern, season_warned = _caught(
            lambda: analyzer.detect_seasonal_pattern("demographic_parity")
        )
        if n_rows == 0:
            assert pattern == {}
            assert season_warned
        else:
            # One observation fills exactly one slot and the docstring says so;
            # it is a per-slot mean, not a cycle verdict, and it invents no value.
            assert pattern == {0: pytest.approx(0.61)}
            assert len(pattern) == 1


# ===========================================================================
# Flag 2: the scope of any_alert
# ===========================================================================


class TestAnyAlertCarriesItsScope:
    def _partial_window(self):
        """Nobody is selected, so the four-fifths ratio has no denominator and is
        REFUSED, while demographic parity measures a real 0.0 and is compared."""
        frame = pd.DataFrame({"prediction": [0] * 100, "group_gender": ["A"] * 50 + ["B"] * 50})
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                min_samples=30, metrics_to_track=["disparate_impact", "demographic_parity"]
            )
        )
        return _caught(lambda: monitor.update_and_check(frame))[0]

    def test_a_false_verdict_over_partial_coverage_says_what_it_did_not_cover(self):
        """REFUSAL PIN. Measured before the fix: any_alert False and
        to_dict()['any_alert'] False, with no key anywhere saying that the
        four-fifths rule had never been applied."""
        snap = self._partial_window()

        assert math.isnan(snap.metrics["disparate_impact_group_gender"])
        assert snap.metrics["demographic_parity_group_gender"] == pytest.approx(0.0)
        assert snap.alerts == {"demographic_parity_group_gender": False}
        # The verdict stays two-valued for the question it answers.
        assert snap.any_alert is False
        # The scope is now beside it, and at the JSON boundary.
        assert snap.uncompared_metrics == ["disparate_impact_group_gender"]
        as_dict = snap.to_dict()
        assert as_dict["n_compared"] == 1
        assert as_dict["uncompared_metrics"] == ["disparate_impact_group_gender"]

    def test_control_a_fully_compared_clean_window_reports_full_coverage(self):
        """OVER-CORRECTION CONTROL, and the reason any_alert must stay two-valued.

        Both metrics are real numbers, both are compared, both are clean. If
        partial coverage turned any_alert into None this window would have to keep
        its False, and it does: the two windows are now distinguishable by the
        coverage keys instead of by the verdict.
        """
        frame = pd.DataFrame({"prediction": [1, 0] * 50, "group_gender": ["A"] * 50 + ["B"] * 50})
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                min_samples=30, metrics_to_track=["disparate_impact", "demographic_parity"]
            )
        )
        snap, warned = _caught(lambda: monitor.update_and_check(frame))

        assert snap.metrics["disparate_impact_group_gender"] == pytest.approx(1.0)
        assert snap.metrics["demographic_parity_group_gender"] == pytest.approx(0.0)
        assert snap.any_alert is False
        assert snap.uncompared_metrics == []
        assert snap.to_dict()["n_compared"] == 2
        assert warned == []

    def test_control_a_custom_metric_does_not_delete_the_false_verdict(self):
        """OVER-CORRECTION CONTROL. A custom metric NEVER gets an alerts entry by
        design, so "None whenever coverage is partial" would turn every clean
        window carrying one into a could-not-check."""
        frame = pd.DataFrame({"prediction": [1, 0] * 50, "group_gender": ["A"] * 50 + ["B"] * 50})
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(
                min_samples=30, metrics_to_track=["disparate_impact", "demographic_parity"]
            ),
            custom_metrics={"my_rate": lambda df: float((df["prediction"] == 1).mean())},
        )
        snap, _ = _caught(lambda: monitor.update_and_check(frame))

        assert snap.metrics["my_rate"] == pytest.approx(0.5)
        assert snap.any_alert is False, "a clean window lost its verdict to a custom metric"
        assert snap.uncompared_metrics == ["my_rate"]

    def test_a_window_that_compared_nothing_is_still_none_not_false(self):
        """The older three-state contract is untouched by the scope keys."""
        frame = pd.DataFrame({"prediction": [1, 0] * 60, "label": [1, 0] * 60})
        monitor = FairnessMonitor(config=FairnessMonitorConfig(min_samples=30))
        snap, _ = _caught(lambda: monitor.update_and_check(frame))

        assert snap.alerts == {}
        assert snap.any_alert is None
        assert snap.to_dict()["any_alert"] is None
        assert snap.to_dict()["n_compared"] == 0
