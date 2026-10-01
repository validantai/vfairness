"""W3: a release-gate bound that CANNOT BE BREACHED, at all four of its sites.

An independent audit overturned twelve PROVEN grades on
``vfairness.operations.cicd.gate`` and reproduced two of them by execution. These
decide whether a model ships, so they are the highest-consequence rows in the
library. One rule closes both:

    A BOUND IS UNUSABLE WHEN THE COMPARISON IT CONTROLS CANNOT BE FALSE FOR ANY
    DATA IN THE METRIC'S RANGE.

That is a property of the bound AND the metric's range together, never of whether
the bound is finite. Infinity is the easiest case of it, and the gate already
refused the infinities: ``_unusable_margin_message`` is gated on
``is_measured(margin)``, which closes NaN and +/-inf and leaves every FINITE
vacuous bound wide open, while its own sentence states the criterion those finite
bounds meet word for word ("the comparison that enforces the requirement is False
whatever the data").

Reproduced on HEAD 2026-09-30 before the fix.

A. A MAXIMUM-POSSIBLE DISPARITY, APPROVED. A model that rejects 100 percent of one
   intersectional cell (f_y) and nobody else, so the intersectional gap is 1.0000,
   the largest a rate disparity can be, against a base threshold of 0.6 relaxed by
   ``default_intersection_threshold_multiplier``::

       mult=1.2    approved=False  'BLOCKED - 3/4 levels passed'   gap 1.0000 vs 0.72  passed=False
       mult=2.0    approved=True   'APPROVED - 4/4 levels passed'  gap 1.0000 vs 1.2   passed=True
       mult=100.0  approved=True   'APPROVED - 4/4 levels passed'  gap 1.0000 vs 60.0  passed=True
       mult=inf    approved=False  'BLOCKED - 3/4 levels passed'   gap 1.0000 vs inf   passed=False

   and at mult 2.0 and 100.0 the report card DROPPED its '### Blocking Issues'
   section entirely, so the page a reviewer merges from said nothing at all about
   the cell that is never selected. ``relax_threshold`` guards ``multiplier <= 0``
   and NaN, so the multiplier carried the bound clean outside the metric's own
   range: a demographic_parity_difference lives in [0, 1] and was compared
   against 60.0.

B. A MEASURED DEGRADATION, APPROVED. Same shape, different parameter: 0.7 against
   a baseline of 0.2, a degradation of 0.5 in the metric's own direction, through
   ``evaluate_from_metrics`` and identically through ``evaluate``::

       improvement_margin        0.01  -> BLOCKED
       improvement_margin        0.0   -> BLOCKED
       improvement_margin       -1.0   -> APPROVED, 0 warnings
       improvement_margin       -1e9   -> APPROVED, 0 warnings
       improvement_margin        inf   -> BLOCKED (the is_measured half)
       allow_degradation_margin  1e9   -> APPROVED, 0 warnings

   ``-1e9`` meets the library's own stated criterion exactly.

THE SIBLINGS, each checked and each covered here:

* the BASE threshold, not only the relaxed one: the same hole is reachable with no
  hierarchy at all, and ``tests/test_bgl4_operations_4.py`` recorded the gate
  approving a 0.90 gap against a bound of 1e9 as EVIDENCE for a separate
  pre-commit finding. That row is corrected in the same change.
* ``per_intersection_thresholds`` overrides, which reach ``evaluate`` as that
  level's own threshold, so one guard covers them.
* BOTH entry points, ``evaluate`` and ``evaluate_from_metrics``, because this file
  has been bitten twice by a fix landing at one of two identical call sites.
* ``operations.cicd.precommit``, which applies this rule to a config FILE at
  commit time and was the only layer applying it at all. Its table and
  ``_metric_direction.UNIT_INTERVAL_METRICS`` are deliberate duplicates (the hook
  must stay stdlib-only), so they are pinned to hold the same names.

Every section carries OVER-CORRECTION CONTROLS asserting a REAL verdict, because a
gate that blocks everything ships nothing and passes every refusal test. A
legitimate loose-but-usable threshold and a legitimate relaxation must still
APPROVE.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    metric_value_range,
    vacuous_bound_reason,
)
from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    ModelFairnessGate,
)

METRIC = "demographic_parity_difference"
INF = float("inf")


def _one_cell_rejected():
    """Four 50-row cells; only ``f_y`` is rejected, so the cell gap is exactly 1.0.

    Every single-attribute group is 100 rows and every cell is 50, so nothing here
    is a small-sample refusal: the only thing that can block is the threshold.
    """
    sex = np.array(["f"] * 100 + ["m"] * 100)
    race = np.array((["x"] * 50 + ["y"] * 50) * 2)
    y_pred = np.array([0 if (s == "f" and r == "y") else 1 for s, r in zip(sex, race)])
    y_true = np.ones(200, dtype=int)
    return y_true, y_pred, sex, race


def _half_of_one_cell_rejected():
    """The same shape with a REAL, sub-maximal cell gap of 0.5, for the controls."""
    y_true, y_pred, sex, race = _one_cell_rejected()
    y_pred = y_pred.copy()
    cell = [i for i, (s, r) in enumerate(zip(sex, race)) if s == "f" and r == "y"]
    for i in cell[: len(cell) // 2]:
        y_pred[i] = 1
    return y_true, y_pred, sex, race


def _gate(**kwargs):
    return ModelFairnessGate(config=GateConfig(metrics=[METRIC], **kwargs))


def _hierarchical(multiplier, *, base_threshold=0.6, data=None):
    y_true, y_pred, sex, race = data or _one_cell_rejected()
    gate = _gate(thresholds={METRIC: base_threshold}, min_group_size=30)
    hconfig = HierarchicalGateConfig(
        default_intersection_threshold_multiplier=multiplier, min_group_size=30
    )
    return gate.evaluate_hierarchical(
        y_true, y_pred, {"sex": sex, "race": race}, hierarchical_config=hconfig
    )


def _intersection_row(decision):
    level = decision.level_results["intersection:sex_x_race"]
    (row,) = level.metric_evaluations
    return row


# ===========================================================================
# A. The relaxed intersectional threshold.
# ===========================================================================


class TestARelaxedThresholdOutsideTheMetricsRange:
    @pytest.mark.parametrize("multiplier,relaxed", [(2.0, 1.2), (100.0, 60.0)])
    def test_the_maximum_possible_disparity_is_not_approved(self, multiplier, relaxed):
        """BEFORE: approved=True, 'APPROVED - 4/4 levels passed', the cell gap
        1.0000 marked passed=True against 1.2 and against 60.0."""
        decision = _hierarchical(multiplier)
        row = _intersection_row(decision)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        # The MEASUREMENT is still published; what is withheld is the verdict.
        assert row.value == pytest.approx(1.0)
        assert row.threshold == pytest.approx(relaxed)
        assert row.passed is False
        assert "COULD NOT CHECK the threshold" in row.message
        assert f"{relaxed:.4f}" in row.message and "[0.0000, 1.0000]" in row.message

    @pytest.mark.parametrize("multiplier", [2.0, 100.0])
    def test_the_report_card_regains_its_blocking_issues_section(self, multiplier):
        """BEFORE: the card for these two multipliers had NO '### Blocking Issues'
        section at all, so the reviewer's page carried nothing about the cell that
        is never selected. The card is where the merge decision is made."""
        decision = _hierarchical(multiplier)
        card = FairnessReportCard(decision, model_name="loan-v3").to_markdown()

        assert "### Blocking Issues" in card
        assert "COULD NOT CHECK the threshold" in card
        assert "lower the multiplier" in card
        # And on the two surfaces a CI integration posts.
        assert "COULD NOT CHECK the threshold" in FairnessReportCard(decision).to_dict()["markdown"]
        assert (
            "COULD NOT CHECK the threshold"
            in FairnessReportCard(decision).to_github_comment_payload()["body"]
        )

    def test_the_hierarchical_markdown_report_names_it_too(self):
        report = _hierarchical(2.0).to_markdown_report()

        assert "## Blocking Issues" in report
        assert "COULD NOT CHECK the threshold" in report

    def test_the_infinite_multiplier_still_refuses_and_reads_the_same(self):
        """The narrow case the gate already refused, kept in the same table so the
        finite and the infinite halves cannot drift apart. relax_threshold leaves
        the bound at inf, and check_threshold refuses an unmeasurable threshold."""
        decision = _hierarchical(INF)
        row = _intersection_row(decision)

        assert decision.approved is False
        assert row.passed is False
        assert row.threshold == INF

    def test_a_per_intersection_override_is_the_same_bound(self):
        """The SIBLING of the multiplier: an override reaches evaluate() as that
        level's threshold, so the one guard covers it. 1.0 is the boundary case,
        because abs(value) > 1.0 is False even for the worst possible gap."""
        y_true, y_pred, sex, race = _one_cell_rejected()
        gate = _gate(thresholds={METRIC: 0.6}, min_group_size=30)
        hconfig = HierarchicalGateConfig(
            min_group_size=30,
            per_intersection_thresholds={"sex_x_race": {METRIC: 1.0}},
        )

        decision = gate.evaluate_hierarchical(
            y_true, y_pred, {"sex": sex, "race": race}, hierarchical_config=hconfig
        )
        row = _intersection_row(decision)

        assert row.threshold == pytest.approx(1.0)
        assert row.passed is False
        assert "COULD NOT CHECK the threshold" in row.message
        assert decision.approved is False

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_legitimately_relaxed_threshold_still_approves(self):
        """The whole point of the multiplier has to keep working. 0.6 x 1.5 = 0.9,
        inside [0, 1] and breachable, and a REAL cell gap of 0.5 passes it: the
        relaxed level approves on the number, not on a bound nobody could fail."""
        decision = _hierarchical(1.5, data=_half_of_one_cell_rejected())
        row = _intersection_row(decision)

        assert row.value == pytest.approx(0.5)
        assert row.threshold == pytest.approx(0.9)
        assert row.passed is True
        assert row.message == ""
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.blocking_reasons == []
        assert "COULD NOT CHECK" not in FairnessReportCard(decision, "m").to_markdown()

    def test_control_the_same_relaxed_bound_still_blocks_a_real_breach(self):
        """The other half of the control: 0.9 is a bound that CAN fail, and the
        maximum disparity fails it. A guard that only ever refuses proves nothing
        about the bound; this shows the same bound produce both verdicts."""
        decision = _hierarchical(1.5)
        row = _intersection_row(decision)

        assert row.value == pytest.approx(1.0)
        assert row.threshold == pytest.approx(0.9)
        assert row.passed is False
        assert "exceeds threshold" in row.message
        assert "COULD NOT CHECK" not in row.message
        assert decision.approved is False

    def test_control_the_default_multiplier_on_a_usual_threshold_is_untouched(self):
        """The documented default: 0.1 x 1.2 = 0.12, which grades. The fix must not
        reach the configuration the docs teach."""
        decision = _hierarchical(1.2, base_threshold=0.1, data=_half_of_one_cell_rejected())
        row = _intersection_row(decision)

        assert row.threshold == pytest.approx(0.12)
        assert row.value == pytest.approx(0.5)
        assert row.passed is False
        assert "exceeds threshold" in row.message


# ===========================================================================
# A-SIBLING. The BASE threshold, with no hierarchy involved.
# ===========================================================================


class TestTheBaseThresholdIsTheSameBound:
    @pytest.mark.parametrize("threshold", [1.0, 1.2, 60.0, 1e9])
    @pytest.mark.parametrize("entry", ["evaluate", "evaluate_from_metrics"])
    def test_a_bound_no_gap_can_exceed_is_refused_at_both_entry_points(self, threshold, entry):
        """BEFORE (recorded in tests/test_bgl4_operations_4.py as evidence for a
        pre-commit finding): a demographic_parity_difference of 0.90 against a
        threshold of 1e9 returned GateDecision(approved, approved=True)."""
        gate = _gate(thresholds={METRIC: threshold})

        if entry == "evaluate_from_metrics":
            decision = gate.evaluate_from_metrics({METRIC: 0.90})
        else:
            protected = np.array(["a"] * 100 + ["b"] * 100)
            y_pred = np.array([1] * 100 + [0] * 90 + [1] * 10)
            decision = gate.evaluate(np.ones(200, dtype=int), y_pred, protected)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert any("COULD NOT CHECK the threshold" in r for r in decision.blocking_reasons)
        assert decision.metric_evaluations[0].value == pytest.approx(0.90)

    def test_control_a_loose_but_usable_threshold_still_approves(self):
        """0.99 is loose, and it is not vacuous: a gap of 1.0 exceeds it. A gate
        configured at 0.99 with a measured 0.5 must still approve, on the number."""
        approved = _gate(thresholds={METRIC: 0.99}).evaluate_from_metrics({METRIC: 0.5})
        blocked = _gate(thresholds={METRIC: 0.99}).evaluate_from_metrics({METRIC: 1.0})

        assert approved.approved is True
        assert approved.status is GateStatus.APPROVED
        assert approved.metric_evaluations[0].passed is True
        assert blocked.approved is False
        assert "exceeds threshold" in blocked.metric_evaluations[0].message

    def test_control_zero_tolerance_is_still_a_real_policy(self):
        """0.0 on a lower-is-better metric is the STRICTEST usable bound, not a
        degenerate one, and check_threshold's own comment says so. It must keep
        passing an exact 0.0 and failing anything above it."""
        clean = _gate(thresholds={METRIC: 0.0}).evaluate_from_metrics({METRIC: 0.0})
        dirty = _gate(thresholds={METRIC: 0.0}).evaluate_from_metrics({METRIC: 0.01})

        assert clean.approved is True
        assert dirty.approved is False
        assert "exceeds threshold" in dirty.metric_evaluations[0].message

    def test_control_the_four_fifths_floor_still_grades(self):
        """The higher-is-better half. 0.80 is the four-fifths rule and it must keep
        working in both directions; 0.0 was already refused by check_threshold and
        still is, with ITS message, not a second one from the new guard."""
        ratio = "disparate_impact_ratio"
        gate = ModelFairnessGate(config=GateConfig(metrics=[ratio], thresholds={ratio: 0.8}))

        assert gate.evaluate_from_metrics({ratio: 0.95}).approved is True
        failed = gate.evaluate_from_metrics({ratio: 0.62})
        assert failed.approved is False
        assert "below the required minimum" in failed.metric_evaluations[0].message

        floor_zero = ModelFairnessGate(
            config=GateConfig(metrics=[ratio], thresholds={ratio: 0.0})
        ).evaluate_from_metrics({ratio: 0.0})
        assert floor_zero.approved is False
        assert "no value can fall below" in floor_zero.metric_evaluations[0].message

    def test_control_a_metric_with_no_declared_range_is_unchanged(self):
        """Three states, never two. Where the range is UNKNOWN no vacuity claim can
        be made, so the bound stays exactly as usable as configured: refusing there
        would block bounds that grade perfectly well (a gap between two continuous
        outcomes is a legitimate magnitude far above 1)."""
        unbounded = "net_benefit_parity"
        assert metric_value_range(unbounded) is None

        decision = ModelFairnessGate(
            config=GateConfig(metrics=[unbounded], thresholds={unbounded: 99.0})
        ).evaluate_from_metrics({unbounded: 3.0})

        assert decision.approved is True
        assert decision.metric_evaluations[0].passed is True


# ===========================================================================
# B. improvement_margin and allow_degradation_margin.
# ===========================================================================


class TestBAMarginNoResultCanFallShortOf:
    DEGRADED = {METRIC: 0.7}
    BASELINE = {METRIC: 0.2}

    def _both_entry_points(self, **config_kwargs):
        """The same configuration through both entry points, on the same numbers.

        evaluate() is handed 200 real rows whose gap is 0.7 (group 'a' selected at
        1.0, group 'b' at 0.3), so the two paths are comparable rather than merely
        adjacent.
        """
        from_metrics = _gate(thresholds={METRIC: 0.9}, **config_kwargs).evaluate_from_metrics(
            self.DEGRADED, baseline_metrics=self.BASELINE
        )
        protected = np.array(["a"] * 100 + ["b"] * 100)
        y_pred = np.array([1] * 100 + [1] * 30 + [0] * 70)
        evaluated = _gate(thresholds={METRIC: 0.9}, **config_kwargs).evaluate(
            np.ones(200, dtype=int), y_pred, protected, baseline_metrics=self.BASELINE
        )
        assert evaluated.metric_evaluations[0].value == pytest.approx(0.7)
        return from_metrics, evaluated

    @pytest.mark.parametrize("margin", [-1.0, -1e9])
    def test_a_required_improvement_nothing_can_fall_short_of_is_refused(self, margin):
        """BEFORE: approved=True with 0 warnings and 0 blocking reasons for a metric
        that got WORSE by 0.5, at both entry points. -1.0 is the exact boundary: the
        largest degradation a [0, 1] metric can suffer is 1.0, so
        improvement < -1.0 is False even for the worst possible result."""
        for decision in self._both_entry_points(
            require_improvement=True, improvement_margin=margin
        ):
            assert decision.approved is False
            assert decision.status is GateStatus.BLOCKED
            (reason,) = [r for r in decision.blocking_reasons if "COULD NOT CHECK" in r]
            assert "required improvement" in reason
            assert "largest possible DEGRADATION is 1.0000" in reason
            # The real degradation is still reported, not replaced by the caveat.
            assert decision.metric_evaluations[0].improvement == pytest.approx(-0.5)

    @pytest.mark.parametrize("margin", [1.0, 1e9])
    def test_an_allowed_degradation_nothing_can_exceed_is_refused(self, margin):
        """The sibling parameter, same rule, same two entry points. BEFORE: 1e9
        approved the same 0.5 degradation with zero warnings."""
        for decision in self._both_entry_points(allow_degradation_margin=margin):
            assert decision.approved is False
            (reason,) = [r for r in decision.blocking_reasons if "COULD NOT CHECK" in r]
            assert "allowed degradation" in reason
            assert "largest possible degradation is 1.0000" in reason

    def test_the_infinite_margins_still_read_as_before(self):
        """The narrow is_measured half, unchanged and NOT doubled up: one refusal
        per bound, in the words of the guard that owns it."""
        for decision in self._both_entry_points(require_improvement=True, improvement_margin=INF):
            (reason,) = [r for r in decision.blocking_reasons if "COULD NOT CHECK" in r]
            assert "its margin is inf, which is not a usable bound" in reason

        for decision in self._both_entry_points(allow_degradation_margin=INF):
            (reason,) = [r for r in decision.blocking_reasons if "COULD NOT CHECK" in r]
            assert "its margin is inf, which is not a usable bound" in reason

    def test_both_vacuous_margins_at_once_are_both_named(self):
        """Two configured requirements, both unusable: the second must not be lost
        behind the first. The threshold is met here (0.7 < 0.9), so neither refusal
        is standing in for a real breach."""
        from_metrics, evaluated = self._both_entry_points(
            require_improvement=True, improvement_margin=-2.0, allow_degradation_margin=5.0
        )
        for decision in (from_metrics, evaluated):
            everything = decision.blocking_reasons + decision.warnings
            assert any("required improvement" in r and "COULD NOT CHECK" in r for r in everything)
            assert any("allowed degradation" in r and "COULD NOT CHECK" in r for r in everything)

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_usable_margin_still_approves_a_real_improvement(self):
        """The requirement has to keep being satisfiable. 0.2 against a baseline of
        0.5 improves by 0.3, past a margin of 0.01, and must APPROVE with the real
        improvement on the row."""
        decision = _gate(
            thresholds={METRIC: 0.9}, require_improvement=True, improvement_margin=0.01
        ).evaluate_from_metrics({METRIC: 0.2}, baseline_metrics={METRIC: 0.5})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.metric_evaluations[0].improvement == pytest.approx(0.3)
        assert decision.warnings == []

    def test_control_a_usable_margin_still_blocks_a_real_shortfall(self):
        """Same margin, worse model: the observed FAIL still reads as a failure and
        not as a could-not-check. The two verdicts have to stay distinguishable."""
        decision = _gate(
            thresholds={METRIC: 0.9}, require_improvement=True, improvement_margin=0.01
        ).evaluate_from_metrics(self.DEGRADED, baseline_metrics=self.BASELINE)

        assert decision.approved is False
        assert decision.blocking_reasons == [f"{METRIC} did not improve by required margin"]
        assert "COULD NOT CHECK" not in " ".join(decision.blocking_reasons)

    @pytest.mark.parametrize(
        "margin,degraded_to,approved",
        [(-0.5, 0.6, True), (-0.5, 0.8, False), (-0.999, 0.7, True)],
    )
    def test_control_a_negative_but_usable_margin_keeps_working(
        self, margin, degraded_to, approved
    ):
        """A NEGATIVE required improvement is legitimate: it tolerates a degradation
        up to its own size. -0.5 is usable because a degradation of 0.6 breaches it,
        and the rule must refuse only the bounds that nothing can breach. -0.999 is
        the last usable value for a [0, 1] metric and is kept usable on purpose."""
        decision = _gate(
            thresholds={METRIC: 0.9}, require_improvement=True, improvement_margin=margin
        ).evaluate_from_metrics({METRIC: degraded_to}, baseline_metrics=self.BASELINE)

        assert decision.approved is approved
        assert "COULD NOT CHECK" not in " ".join(decision.blocking_reasons)

    @pytest.mark.parametrize(
        "margin,degraded_to,approved", [(0.5, 0.3, True), (0.5, 0.8, False), (0.0, 0.2, True)]
    )
    def test_control_a_usable_degradation_bound_keeps_working(self, margin, degraded_to, approved):
        """0.5 allows half the range and refuses more than that; 0.0 is zero
        tolerance, which an UNCHANGED metric still passes. Both are real bounds and
        neither may be swept up by the vacuity rule."""
        decision = _gate(
            thresholds={METRIC: 0.9}, allow_degradation_margin=margin
        ).evaluate_from_metrics({METRIC: degraded_to}, baseline_metrics=self.BASELINE)

        assert decision.approved is approved
        assert "COULD NOT CHECK" not in " ".join(decision.blocking_reasons)

    def test_control_an_unconfigured_requirement_is_still_not_a_requirement(self):
        """With require_improvement off and no degradation bound, NOTHING was
        required, so no bound can be unusable and the verdict rests on the
        threshold alone. A guard that refused every gate without a baseline
        requirement would be the over-correction."""
        decision = _gate(thresholds={METRIC: 0.9}).evaluate_from_metrics(
            self.DEGRADED, baseline_metrics=self.BASELINE
        )

        assert decision.approved is True
        assert decision.blocking_reasons == []
        assert decision.warnings == []


# ===========================================================================
# THE PREDICATE ITSELF, and the sibling table it has to agree with.
# ===========================================================================


class TestTheRuleIsOneTestUsedEverywhere:
    @pytest.mark.parametrize(
        "role,bound,vacuous",
        [
            (BoundRole.THRESHOLD, 0.99, False),
            (BoundRole.THRESHOLD, 1.0, True),
            (BoundRole.THRESHOLD, 60.0, True),
            (BoundRole.REQUIRED_IMPROVEMENT, -0.999, False),
            (BoundRole.REQUIRED_IMPROVEMENT, -1.0, True),
            (BoundRole.ALLOWED_DEGRADATION, 0.999, False),
            (BoundRole.ALLOWED_DEGRADATION, 1.0, True),
        ],
    )
    def test_it_can_say_both_yes_and_no_at_the_boundary(self, role, bound, vacuous):
        """A predicate that answered one way for everything would pass every
        refusal test in this file. Both answers, from the same call, one step
        apart."""
        assert (vacuous_bound_reason(METRIC, bound, role) is not None) is vacuous

    @pytest.mark.parametrize("bound", [float("nan"), INF, -INF, None, True, "0.5"])
    def test_a_non_measurement_is_left_to_the_guard_that_owns_it(self, bound):
        """One refusal per bound. The is_measured guards at each call site already
        refuse these, and answering here as well would put two different
        could-not-check sentences on one configuration value."""
        assert vacuous_bound_reason(METRIC, bound, BoundRole.THRESHOLD) is None

    def test_the_two_range_tables_hold_the_same_names(self):
        """``precommit`` applies this rule to a config FILE and must not disagree
        with the runtime gate about which metrics have a known range. The
        duplication is deliberate (the hook stays stdlib-only, so it cannot import
        _triage); the DIVERGENCE would not be. Measured 2026-09-30: precommit
        carried 8 names the runtime table lacked and the runtime table carried 7 it
        lacked, so a bound of 1e9 on auroc_parity passed the hook."""
        from vfairness.evaluation.vfairness_metrics._metric_direction import (
            UNIT_INTERVAL_METRICS,
        )
        from vfairness.operations.cicd.precommit import (
            MIN_MAX_RATIO_METRICS,
            RATE_DIFFERENCE_METRICS,
        )

        hook = RATE_DIFFERENCE_METRICS | MIN_MAX_RATIO_METRICS
        assert hook == UNIT_INTERVAL_METRICS, {
            "only in the pre-commit hook": sorted(hook - UNIT_INTERVAL_METRICS),
            "only in the runtime rule": sorted(UNIT_INTERVAL_METRICS - hook),
        }

    @pytest.mark.parametrize(
        "metric,bound,expected_in_output",
        [
            ("auroc_parity", 1e9, "no value can exceed this bound"),
            ("worst_group_accuracy", 0.0, "every possible value meets it"),
            ("demographic_parity_ratio", 5.0, "no value can reach this floor"),
        ],
    )
    def test_the_hook_now_refuses_the_names_it_was_missing(
        self, tmp_path, capsys, metric, bound, expected_in_output
    ):
        from vfairness.operations.cicd.precommit import check_fairness_config

        path = tmp_path / "fairness.json"
        path.write_text(json.dumps({"metrics": [metric], "thresholds": {metric: bound}}))

        assert check_fairness_config([str(path)]) == 1
        printed = capsys.readouterr().out
        assert "FAIL" in printed and expected_in_output in printed, printed

    def test_control_the_hook_still_passes_a_working_config_for_those_names(self, tmp_path, capsys):
        """The other half: the names added to the hook's tables must not make a
        real configuration fail. 0.05 on a parity gap and 0.80 on an accuracy floor
        are both usable bounds."""
        from vfairness.operations.cicd.precommit import check_fairness_config

        path = tmp_path / "fairness.json"
        path.write_text(
            json.dumps(
                {
                    "metrics": ["auroc_parity", "worst_group_accuracy"],
                    "thresholds": {"auroc_parity": 0.05, "worst_group_accuracy": 0.80},
                }
            )
        )

        assert check_fairness_config([str(path)]) == 0, capsys.readouterr().out


# ===========================================================================
# THE FOURTH BOUND, found by asking what the sibling was: min_group_size.
# ===========================================================================


class TestAGroupSizeMinimumNoCountCanFallBelow:
    """``_undersized_groups`` tests ``count < min_group_size`` over groups numpy
    found, so every group holds at least one row and the test is False for every
    group of every dataset once the minimum is 1 or less. THE THRESHOLD OF VACUITY
    IS 1, NOT 0, and it is the identical off-by-one that
    ``_grouping.GroupManager.get_invalid_groups`` closed on 2026-09-27, whose own
    comment names ``operations/cicd/gate`` as a production call site. That fix
    covered the METRIC layer's parameter; the gate's own certification minimum was
    never tested for it.

    Measured on HEAD 2026-09-30, one group of 200 and one group of ONE person, a
    selection-rate gap of 0.5000 against a threshold of 0.9::

        min_group_size=30  approved=False  warnings [('b', 1)]  check_ran True
        min_group_size=2   approved=False  warnings [('b', 1)]  check_ran True
        min_group_size=1   approved=True   warnings []          check_ran True
        min_group_size=0   approved=True   warnings []          check_ran True
        min_group_size=-5  approved=True   warnings []          check_ran True

    and at 1, 0 and -5 the card read "RAN, and every group was at or above the
    minimum. This is a measurement, not an absence of one." about a comparison that
    could not have failed.
    """

    @staticmethod
    def _one_person_group():
        y_pred = np.array([1] * 100 + [0] * 100 + [1])
        protected = np.array(["a"] * 200 + ["b"])
        return np.ones(201, dtype=int), y_pred, protected

    @pytest.mark.parametrize("minimum", [1, 0, -5])
    def test_the_empty_warning_list_is_not_published_as_a_measurement(self, minimum):
        y_true, y_pred, protected = self._one_person_group()

        decision = _gate(thresholds={METRIC: 0.9}, min_group_size=minimum).evaluate(
            y_true, y_pred, protected
        )

        assert decision.small_sample_warnings == []
        assert decision.small_sample_check_ran is False
        assert decision.small_sample_vacuous_minimum == minimum
        assert decision.to_dict()["small_sample_vacuous_minimum"] == minimum

    @pytest.mark.parametrize("minimum", [1, 0])
    @pytest.mark.parametrize("surface", ["card", "card_dict", "github_payload", "report"])
    def test_every_surface_names_the_bound_that_could_not_fire(self, minimum, surface):
        y_true, y_pred, protected = self._one_person_group()
        decision = _gate(thresholds={METRIC: 0.9}, min_group_size=minimum).evaluate(
            y_true, y_pred, protected
        )
        card = FairnessReportCard(decision, model_name="loan-v3")
        texts = {
            "card": card.to_markdown(),
            "card_dict": card.to_dict()["markdown"],
            "github_payload": card.to_github_comment_payload()["body"],
            "report": decision.to_markdown_report(),
        }

        text = texts[surface]
        assert f"the minimum was {minimum}" in text
        assert "no group count could fall below it" in text
        assert "The smallest minimum that can flag anything is 2" in text
        # The cause that is NOT true of this decision: it HAS the group sizes.
        assert "no group sizes attached" not in text
        assert "RAN, and every group was at or above the minimum" not in text

    def test_the_hierarchy_applies_the_same_rule_to_its_own_bound(self):
        """The SIBLING call site. evaluate_hierarchical scans with
        hconfig.min_group_size, which is a different parameter from the gate's."""
        y_true, y_pred, protected = self._one_person_group()
        gate = _gate(thresholds={METRIC: 0.9}, min_group_size=30)

        decision = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"sex": protected},
            hierarchical_config=HierarchicalGateConfig(check_intersections=False, min_group_size=1),
        )

        assert decision.small_sample_warnings == []
        assert decision.small_sample_check_ran is False
        assert decision.small_sample_vacuous_minimum == 1
        assert decision.to_dict()["small_sample_vacuous_minimum"] == 1
        for text in (
            decision.to_markdown_report(),
            FairnessReportCard(decision, "m").to_markdown(),
        ):
            assert "the minimum was 1" in text
            assert "check_single_attributes is off" not in text

    # ---- OVER-CORRECTION CONTROLS ----

    @pytest.mark.parametrize("minimum", [2, 30])
    def test_control_a_minimum_that_can_fire_still_fires(self, minimum):
        """2 is the smallest usable minimum and it must keep naming the one-person
        group, refusing the comparison, and reading as a measurement."""
        y_true, y_pred, protected = self._one_person_group()

        decision = _gate(thresholds={METRIC: 0.9}, min_group_size=minimum).evaluate(
            y_true, y_pred, protected
        )

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 1)]
        assert decision.small_sample_check_ran is True
        assert decision.small_sample_vacuous_minimum is None
        assert decision.approved is False
        for text in (
            decision.to_markdown_report(),
            FairnessReportCard(decision, "m").to_markdown(),
        ):
            assert "1 samples" in text
            assert "could not fall below it" not in text

    def test_control_the_verdict_and_the_measurement_are_left_alone(self):
        """DELIBERATELY NOT BLOCKED. A minimum of 1 is the only way a caller can say
        "do not certify group sizes", so refusing it would leave no way to say that,
        and a guard that argues for breaking the product is the wrong guard. What
        changes is the CLAIM, not the verdict: the gap is still measured and
        reported, and the threshold still decides."""
        y_true, y_pred, protected = self._one_person_group()

        decision = _gate(thresholds={METRIC: 0.9}, min_group_size=1).evaluate(
            y_true, y_pred, protected
        )

        assert decision.approved is True
        assert decision.metric_evaluations[0].value == pytest.approx(0.5)
        assert decision.metric_evaluations[0].passed is True
        assert decision.blocking_reasons == []

        # And the threshold still decides: the same data against a bound it breaches.
        blocked = _gate(thresholds={METRIC: 0.1}, min_group_size=1).evaluate(
            y_true, y_pred, protected
        )
        assert blocked.approved is False
        assert "exceeds threshold" in blocked.metric_evaluations[0].message

    def test_control_the_default_minimum_is_untouched(self):
        """DEFAULT_MIN_GROUP_SIZE is 30, so no configuration the library ships
        reaches this disclosure, and a clean run still reads as a measurement."""
        y = np.array([1, 0] * 100)
        protected = np.array(["a"] * 100 + ["b"] * 100)

        decision = _gate(thresholds={METRIC: 0.9}).evaluate(y, y, protected)

        assert decision.small_sample_check_ran is True
        assert decision.small_sample_vacuous_minimum is None
        assert "RAN, and every group was at or above the minimum" in decision.to_markdown_report()
