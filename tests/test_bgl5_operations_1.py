"""BGL5 batch A-operations-1: the eleven overturned grades, closed and pinned.

An auditor overturned eleven of the twenty-six grades in this batch and proved each
one by execution. ``tests/test_bgl4_operations_1.py`` holds the auditor's twelve
demonstrations, which XPASSed the moment these fixes landed and are now plain pins.
THIS file holds the other half of the evidence: the refusal side of each fix, and for
every one of them an OVER-CORRECTION CONTROL asserting that healthy input still gets
its real measurement, with the actual number, because a fix that refuses everything
passes every refusal test and destroys the library.

Five root causes, measured on 2026-09-27:

1. ``ModelFairnessGate.evaluate`` and ``.evaluate_from_metrics`` approved a release
   from a NaN BASELINE. With ``require_improvement=True`` and
   ``baseline_metrics={metric: nan}``: approved=True, "APPROVED - All fairness
   requirements met", improvement=nan, markdown "Pass", GitHub check "success".
   ``improvement < margin`` is False for NaN and ``degradation > margin`` is False for
   NaN, so ONE unguarded baseline silently switched off BOTH requirements. The metric
   VALUE was guarded three ways; the baseline it is compared against was not guarded
   at all. Fixed with one guard ABOVE both checks, at both call sites, through one
   shared message builder.

2. ``evaluate_hierarchical`` built its refusal reason into a LOCAL list and
   ``IntersectionalGateDecision`` had no field for one, so ``to_dict``,
   ``to_markdown_report``, ``FairnessReportCard`` and the GitHub comment payload
   carried none of it: ``evaluate_hierarchical(y, y, {})`` returned BLOCKED, the FAIL
   state, for a could-not-check, and 'nothing was checked' appeared on no surface.

3. ``FairnessReportCard._format_hierarchical`` printed no reasons section at all (0 of
   22 body lines were executed by the test named as that grade's evidence), so a
   never-computed metric rendered as "Failed Metrics: nan" with none of the disclosure
   the flat half carries. ``IntersectionalGateDecision.to_markdown_report`` had the
   same gap. Both now mirror the flat renderers, which is the property the named test
   already asserted for the flat branch.

4. ``GateDecision.to_dict`` emitted ``small_sample_warnings: []`` for a check that ran
   and found nothing AND for a check that never ran, with no key naming which,
   although ``evaluate_from_metrics``' own docstring insists on the difference.

5. ``build_drift_event`` invented ``regulatory_risk`` 0.5, ``population_impact`` 0.3,
   ``drift_velocity`` 0.3 and ``historical_discrimination`` 0.5, which carried an
   alert to (5.1, 'HIGH') and into slack while the same context supplied honestly
   scored (nan, 'UNSCORED'). The unit's own docstring recorded it as "DEFECT NOT
   CLOSED, disclosed only".

6. The Tier-3 ``create_technical_view`` was the third rendering of two quantities
   fixed on the other two tiers and had neither fix, and ``create_drift_timeline``
   said nothing at all about the commonest unmeasured input, an EMPTY drift table.

The last section re-aims two sabotage records the auditor proved stale: the recorded
single-branch sabotage of ``_drift_coverage_note`` left 32 tests passing, because the
other branch answers first for the fixture store. Each branch is now pinned on its
own, so either sabotage alone reddens.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    GateStatus,
    ModelFairnessGate,
)
from vfairness.operations.monitoring.alerts import FairnessAlertPrioritizer
from vfairness.operations.reporting.dashboard import (
    DashboardConfig,
    FairnessDashboard,
    _drift_coverage_note,
)
from vfairness.operations.reporting.store import HealthScore, MetricsStore

CFG = DashboardConfig()
METRIC = "demographic_parity_difference"
NAN = float("nan")


# ===== fixtures


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*args, **kwargs)


def _caught(fn, *args, **kwargs):
    """Run *fn* with warnings enabled, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in record]


def _gate(**kwargs):
    return ModelFairnessGate(
        config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.5}, **kwargs)
    )


def _fair_arrays(n: int = 100):
    """Two groups of *n* rows with identical selection rates, so the gap is 0.0."""
    y = np.array([1, 0] * n)
    protected = np.array(["a"] * n + ["b"] * n)
    return y, y, protected


def _store(groups, values, determined=None):
    store = MetricsStore()
    t0 = datetime.now() - timedelta(days=2)
    n = len(groups)
    store.ingest_dataframe(
        pd.DataFrame(
            {
                "timestamp": [t0 + timedelta(hours=i) for i in range(n)],
                "metric": [METRIC] * n,
                "value": values,
                "group": groups,
                "alert": [False] * n,
                "alert_determined": determined if determined is not None else [True] * n,
                "group_size": [500] * n,
            }
        ),
        group_col="group",
        alert_col="alert",
        alert_determined_col="alert_determined",
        group_size_col="group_size",
    )
    return store


def _clean_store():
    """Four measured, threshold-compared, non-breaching records. No drift row."""
    return _store(
        ["gender_male", "gender_female", "gender_male", "gender_female"],
        [0.02, 0.03, 0.01, 0.04],
    )


class _DriftResult:
    """The shape MetricsStore.ingest_drift_result reads."""

    def __init__(self, score=0.05, detected=False, hours_ago=1):
        self.timestamp = datetime.now() - timedelta(hours=hours_ago)
        self.overall_drift_score = score
        self.drift_detected = detected
        self.metric = METRIC
        self.mmd_score = 0.0
        self.worst_scale = None


def _texts(fig):
    return [a.text or "" for a in (fig.layout.annotations or ())]


def _severity_bar(fig):
    return [t for t in fig.data if t.type == "bar" and "LOW" in [str(x) for x in t.x]][0]


def _health(components):
    """A HealthScore carrying exactly *components*, for the coverage-note branches."""
    return HealthScore(
        score=100.0,
        status="healthy",
        trend="stable",
        trend_slope=0.0,
        components=components,
        timestamp=datetime.now(),
        explanation="fixture",
    )


# ===========================================================================
# CAUSE 1. A baseline requirement that could not be checked is not a met one.
# ===========================================================================


class TestTheGateRefusesABaselineItCouldNotMeasure:
    def test_evaluate_fails_closed_on_a_nan_baseline_it_was_told_to_require(self):
        """BEFORE: approved=True, APPROVED, 'All fairness requirements met',
        improvement=nan, markdown '| ... | 0.0000 | 0.5000 | Pass |', GitHub
        conclusion 'success'."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)
        y_true, y_pred, protected = _fair_arrays()

        decision = gate.evaluate(y_true, y_pred, protected, baseline_metrics={METRIC: NAN})
        row = decision.metric_evaluations[0]

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert row.passed is False
        # The measured value is still reported. Only the VERDICT is withheld.
        assert row.value == pytest.approx(0.0)
        assert math.isnan(row.improvement), row.improvement
        assert "baseline" in row.message and "never checked" in row.message
        assert any("baseline" in r for r in decision.blocking_reasons)
        assert gate.create_github_check(decision)["conclusion"] == "failure"
        assert "Pass" not in decision.to_markdown_report().split("## Blocking Issues")[0]

    def test_evaluate_from_metrics_fails_closed_on_the_same_hole(self):
        """The entry point a CI system hands pre-computed numbers to. BEFORE:
        approved=True, APPROVED, row {'value': 0.02, 'passed': True,
        'baseline_value': nan, 'improvement': nan}, check 'success'."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics({METRIC: 0.02}, baseline_metrics={METRIC: NAN})

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert decision.metric_evaluations[0].value == pytest.approx(0.02)
        assert any("could not be measured" in r for r in decision.blocking_reasons)
        assert gate.create_github_check(decision)["conclusion"] == "failure"

    def test_an_infinite_baseline_is_refused_on_the_same_guard(self):
        """NaN loses every comparison and inf WINS them: they are two halves of
        one hole, which is why the guard asks is_measured rather than isnan."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics(
            {METRIC: 0.02}, baseline_metrics={METRIC: float("inf")}
        )

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED

    def test_the_degradation_bound_alone_is_enough_to_require_the_baseline(self):
        """allow_degradation_margin is the OTHER requirement the same NaN
        switched off, and it is configured independently of
        require_improvement. The guard sits above both, so either one on its own
        makes an unmeasurable baseline a could-not-check."""
        gate = _gate(allow_degradation_margin=0.0)

        decision = gate.evaluate_from_metrics({METRIC: 0.09}, baseline_metrics={METRIC: NAN})

        assert decision.approved is False
        assert any("baseline" in r for r in decision.blocking_reasons)

    def test_a_non_blocking_metric_downgrades_instead_of_blocking(self):
        """Three states, not two: a non-blocking metric whose baseline could not
        be measured is CONDITIONAL with the reason in warnings, and GitHub's
        'neutral', never a green tick and never a block."""
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=[METRIC],
                thresholds={METRIC: 0.5},
                require_improvement=True,
                improvement_margin=0.01,
                blocking_metrics=[],
            )
        )

        decision = gate.evaluate_from_metrics({METRIC: 0.02}, baseline_metrics={METRIC: NAN})

        assert decision.approved is True
        assert decision.status is GateStatus.CONDITIONAL
        assert any("baseline" in w for w in decision.warnings)
        assert gate.create_github_check(decision)["conclusion"] == "neutral"

    def test_a_threshold_breach_keeps_its_own_more_precise_reason(self):
        """A metric already failing on the comparison must not have its reason
        replaced by the baseline one; the unchecked requirement is still
        reported, as a warning beside it."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics({METRIC: 0.9}, baseline_metrics={METRIC: NAN})

        assert decision.approved is False
        assert "exceeds threshold" in decision.blocking_reasons[0]
        assert any("baseline" in w for w in decision.warnings)

    # OVER-CORRECTION CONTROLS

    def test_control_a_measured_baseline_still_produces_its_real_improvement(self):
        """The measured path is untouched: a real baseline of 0.5000 against a
        measured 0.0000 gap is an improvement of exactly 0.5, and the gate
        approves with a green check."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)
        y_true, y_pred, protected = _fair_arrays()

        decision = gate.evaluate(y_true, y_pred, protected, baseline_metrics={METRIC: 0.5})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.metric_evaluations[0].improvement == pytest.approx(0.5)
        assert gate.create_github_check(decision)["conclusion"] == "success"

    def test_control_a_real_failure_to_improve_still_blocks(self):
        """The requirement still bites when it CAN be checked: 0.02 against a
        baseline of 0.021 improves by 0.001, under the 0.01 margin."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics({METRIC: 0.02}, baseline_metrics={METRIC: 0.021})

        assert decision.approved is False
        assert any("did not improve" in r for r in decision.blocking_reasons)

    def test_control_a_nan_baseline_with_nothing_required_changes_no_verdict(self):
        """THE over-correction control for this cause. With
        require_improvement=False and allow_degradation_margin=None nothing was
        REQUIRED of the baseline, so nothing went unchecked: the gate still
        approves, reports the measured 0.0 gap, and leaves improvement at None
        (no comparison was asked for), not NaN."""
        gate = _gate()
        y_true, y_pred, protected = _fair_arrays()

        decision = gate.evaluate(y_true, y_pred, protected, baseline_metrics={METRIC: NAN})
        row = decision.metric_evaluations[0]

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert row.value == pytest.approx(0.0)
        assert row.improvement is None
        assert gate.create_github_check(decision)["conclusion"] == "success"


# ===========================================================================
# CAUSES 2 and 3. The hierarchical decision and every renderer of it.
# ===========================================================================


def _unmeasurable_hierarchy():
    """One group in the only attribute, so no disparity is definable anywhere."""
    y_true, y_pred, _ = _fair_arrays()
    return _gate().evaluate_hierarchical(y_true, y_pred, {"gender": np.array(["a"] * 200)})


def _measurable_hierarchy():
    y_true, y_pred, protected = _fair_arrays()
    return _gate().evaluate_hierarchical(y_true, y_pred, {"gender": protected})


class TestTheHierarchicalRefusalReachesItsConsumers:
    def test_the_zero_level_refusal_is_on_the_dict_the_markdown_and_the_card(self):
        """BEFORE: to_dict had no blocking_reasons key at all, the markdown
        report ended at '**Levels evaluated**: 0' and the card said
        'Deployment blocked', so the sentence the method appends for itself was
        readable nowhere."""
        decision = _gate().evaluate_hierarchical(*_fair_arrays()[:2], {})
        payload = decision.to_dict()

        assert decision.approved is False
        assert payload["blocking_reasons"], payload
        assert "nothing was checked" in " ".join(payload["blocking_reasons"])
        assert "no level was evaluated" in decision.to_markdown_report()
        assert "nothing was checked" in FairnessReportCard(decision, "m").to_markdown()
        # And through the JSON round trip a CI integration actually posts.
        assert "nothing was checked" in json.dumps(payload, default=str)

    def test_an_uncomputable_metric_says_so_on_the_pull_request_comment(self):
        """BEFORE: '#### overall: Failed Metrics / - **demographic_parity_
        difference**: nan (threshold: 0.5000)' and the strings 'could not be
        computed', 'fails closed', 'not measur' and 'could-not-check' were ALL
        absent from the whole card."""
        card = FairnessReportCard(_unmeasurable_hierarchy(), model_name="loan-v3")
        markdown = card.to_markdown()

        assert "### Blocking Issues" in markdown
        assert "could not be computed" in markdown
        assert "fails closed" in markdown
        # The three graded serialisers all carry the same body.
        assert "could not be computed" in card.to_dict()["markdown"]
        assert "could not be computed" in card.to_github_comment_payload()["body"]

    def test_the_hierarchical_markdown_report_mirrors_the_flat_one(self):
        """GateDecision.to_markdown_report has carried a Blocking Issues section
        all along; this renderer had none, so 'nan | 0.5000 | Fail' was the whole
        story a reader got for a metric nobody could compute."""
        report = _unmeasurable_hierarchy().to_markdown_report()

        assert "## Blocking Issues" in report
        assert "could not be computed" in report

    def test_every_declared_field_still_reaches_the_dict(self):
        """The 'copy constructor drops every later field' shape, which is how
        summary_decision was lost once already: two fields were added to this
        dataclass and the serialiser has to carry both."""
        import dataclasses

        decision = _unmeasurable_hierarchy()
        declared = {f.name for f in dataclasses.fields(decision)}

        assert declared <= set(decision.to_dict()), sorted(declared - set(decision.to_dict()))

    def test_the_conditional_arm_carries_its_warnings_to_both_renderers(self):
        """The THIRD state on this path, not just the fail one. A NON-BLOCKING
        metric that could not be computed makes the hierarchy CONDITIONAL rather
        than BLOCKED, and that sentence was dropped at the same constructor:
        measured on metrics=[dp, 'auroc_parity'] (auroc_parity is not one of the
        five evaluate() can compute) with blocking_metrics=[dp] over two real
        groups of 100, the decision is approved=True, CONDITIONAL, 2 warnings."""
        y_true, y_pred, protected = _fair_arrays()
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=[METRIC, "auroc_parity"],
                thresholds={METRIC: 0.5, "auroc_parity": 0.1},
                blocking_metrics=[METRIC],
            )
        )

        decision = gate.evaluate_hierarchical(y_true, y_pred, {"gender": protected})

        assert decision.approved is True
        assert decision.status is GateStatus.CONDITIONAL
        assert decision.blocking_reasons == []
        assert any("auroc_parity could not be computed" in w for w in decision.warnings)
        assert "## Warnings" in decision.to_markdown_report()
        assert "### Warnings" in FairnessReportCard(decision, "m").to_markdown()

    # OVER-CORRECTION CONTROL

    def test_control_a_measurable_hierarchy_reports_no_reason_at_all(self):
        """Two real groups of 100 at identical selection rates: every level is
        measurable and passes, so the decision is APPROVED with 2/2 levels, an
        EMPTY blocking_reasons list, and no Blocking Issues section on any
        surface. A renderer that always printed the section would be the
        over-correction."""
        decision = _measurable_hierarchy()

        assert decision.approved is True
        assert decision.summary == "APPROVED - 2/2 levels passed"
        assert decision.blocking_reasons == []
        assert decision.to_dict()["blocking_reasons"] == []
        assert "Blocking Issues" not in decision.to_markdown_report()
        assert "Blocking Issues" not in FairnessReportCard(decision, "m").to_markdown()
        assert list(decision.level_results) == ["overall", "attr:gender"]


# ===========================================================================
# CAUSE 4. An empty small-sample list means one of two things, and said so.
# ===========================================================================


class TestAnUncheckedSmallSampleListSaysThatItIsUnchecked:
    def test_the_dict_distinguishes_never_checked_from_checked_and_clean(self):
        """BEFORE, measured: both decisions serialised small_sample_warnings []
        with 'any key naming whether the check RAN' equal to
        ['small_sample_warnings'], i.e. none."""
        gate = _gate()
        y_true, y_pred, protected = _fair_arrays()

        checked = gate.evaluate(y_true, y_pred, protected).to_dict()
        never_checked = gate.evaluate_from_metrics({METRIC: 0.02}).to_dict()

        assert checked["small_sample_warnings"] == []
        assert never_checked["small_sample_warnings"] == []
        assert checked["small_sample_check_ran"] is True
        assert never_checked["small_sample_check_ran"] is False

    def test_the_markdown_report_names_the_check_that_did_not_run(self):
        """The markdown is what create_github_check publishes as output.text, so
        the distinction has to be legible there too, not only in JSON."""
        report = _gate().evaluate_from_metrics({METRIC: 0.02}).to_markdown_report()

        assert "## Small-Sample Check" in report
        assert "NOT RUN" in report
        assert "did not happen" in report

    def test_a_hand_built_decision_claims_neither(self):
        """Three states: True, False and 'nobody recorded'. A GateDecision built
        by a third party has not established that the check was skipped, only
        that nothing says so."""
        from vfairness.operations.cicd.gate import GateDecision, MetricEvaluation

        decision = GateDecision(
            approved=True,
            status=GateStatus.APPROVED,
            metric_evaluations=[
                MetricEvaluation(metric_name=METRIC, value=0.02, threshold=0.1, passed=True)
            ],
        )

        assert decision.to_dict()["small_sample_check_ran"] is None
        assert "NOT RECORDED" in decision.to_markdown_report()

    # OVER-CORRECTION CONTROLS

    def test_control_the_checked_and_clean_decision_is_not_qualified(self):
        """evaluate() HAS the group labels, so its empty list is a real finding
        and must not carry a could-not-check note. Subject unchanged.

        Mechanism updated in W3 (2026-09-30): the report used to print NOTHING for
        the True state, while FairnessReportCard printed the measurement for the
        same decision, so the two surfaces answered one question differently.
        Both state it now, and "not qualified" is asserted as the absence of the
        caveat rather than the absence of the section.
        """
        y_true, y_pred, protected = _fair_arrays()
        decision = _gate().evaluate(y_true, y_pred, protected)
        report = decision.to_markdown_report()

        assert decision.small_sample_check_ran is True
        assert "RAN, and every group was at or above the minimum" in report
        for caveat in ("NOT RUN", "NOT RECORDED", "NOT SIZED", "cannot be read"):
            assert caveat not in report, caveat
        assert decision.approved is True

    def test_control_a_real_small_group_still_reports_its_size(self):
        """The check that DID run still names the five-person group and still
        refuses to certify the comparison drawn across it."""
        y = np.array([1, 0] * 50 + [1, 0, 1, 0, 1])
        protected = np.array(["a"] * 100 + ["b"] * 5)
        decision = _gate().evaluate(y, y, protected)

        assert decision.approved is False
        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 5)]
        assert decision.to_dict()["small_sample_check_ran"] is True
        assert "## Small-Sample Warnings" in decision.to_markdown_report()


# ===========================================================================
# CAUSE 5. build_drift_event invented the four factors that decide the severity.
# ===========================================================================


class TestBuildDriftEventNoLongerSuppliesItsOwnContext:
    def test_the_unsupplied_factors_are_absent_and_named(self):
        """BEFORE: the event carried regulatory_risk 0.5, population_impact 0.3,
        drift_velocity 0.3 and historical_discrimination 0.5, which
        calculate_priority cannot tell from measurements."""
        event, caught = _caught(FairnessAlertPrioritizer.build_drift_event, METRIC, ["B"], 0.5, 0.1)

        for factor in FairnessAlertPrioritizer._CONTEXT_FACTORS:
            assert factor not in event, (factor, event)
            assert factor in caught[0], (factor, caught)
        assert len(caught) == 1
        # The measurements the caller DID supply are still there, unchanged.
        assert event["drift_score"] == pytest.approx(0.5)
        assert event["mean_shift"] == pytest.approx(0.1)
        assert event["affected_groups"] == ["B"]

    def test_both_routes_to_the_same_unmeasured_context_now_agree(self):
        """BEFORE: (5.1, 'HIGH') through the helper and (nan, 'UNSCORED') through
        a hand-built dict, for the same unmeasured context. 5.1 cleared the 5.0
        HIGH band on the substitutes alone."""
        prioritizer = FairnessAlertPrioritizer()
        built, _ = _caught(FairnessAlertPrioritizer.build_drift_event, METRIC, ["B"], 0.5, 0.1)

        through_helper = _quiet(prioritizer.calculate_priority, built)
        by_hand = _quiet(prioritizer.calculate_priority, {"drift_score": 0.5, "mean_shift": 0.1})

        assert math.isnan(through_helper[0]) and through_helper[1] == "UNSCORED"
        assert math.isnan(by_hand[0]) and by_hand[1] == "UNSCORED"

    def test_the_alert_built_from_it_is_unscored_and_routed_for_hand_triage(self):
        """BEFORE: severity HIGH, priority 5.10, route {'channel': 'slack',
        'team': '#fairness-alerts'} and a message reading '[HIGH] ... Priority
        score: 5.10' with nothing in it about the four invented numbers."""
        prioritizer = FairnessAlertPrioritizer()
        event = _quiet(FairnessAlertPrioritizer.build_drift_event, METRIC, ["B"], 0.5, 0.1)

        alert = _quiet(prioritizer.create_alert, event)

        assert alert.severity == prioritizer.UNSCORED_SEVERITY
        assert math.isnan(alert.priority_score)
        assert "triage" in alert.routing
        assert alert.message.startswith("[UNSCORED]")

    # OVER-CORRECTION CONTROLS

    def test_control_a_fully_supplied_event_still_scores_its_real_number(self):
        """All four factors measured: 1.0*3.5 + 0.8*2.0 + 0.9*2.5 + 1.0*3.0 =
        10.35, plus the 0.82 drift boost and the 0.5 intersectional boost =
        11.67, CRITICAL, and not one warning."""
        event, caught = _caught(
            FairnessAlertPrioritizer.build_drift_event,
            "equalized_odds",
            ["Black", "Female"],
            0.82,
            0.07,
            regulatory_risk=1.0,
            population_impact=0.8,
            drift_velocity=0.9,
            historical_discrimination=1.0,
            intersectional=True,
        )
        (score, severity), warned = _caught(FairnessAlertPrioritizer().calculate_priority, event)

        assert caught == []
        assert warned == []
        assert score == pytest.approx(11.67, abs=0.01)
        assert severity == "CRITICAL"

    def test_control_a_partly_supplied_event_scores_on_what_was_measured(self):
        """Refusing everything would be the over-correction. Two factors
        supplied, regulatory_risk 1.0*3.5 + historical_discrimination 1.0*3.0 =
        6.5, plus the 0.5 drift boost = 7.0, HIGH, with the other two named."""
        event, _ = _caught(
            FairnessAlertPrioritizer.build_drift_event,
            METRIC,
            ["B"],
            0.5,
            0.1,
            regulatory_risk=1.0,
            historical_discrimination=1.0,
        )
        score, severity = _quiet(FairnessAlertPrioritizer().calculate_priority, event)

        assert score == pytest.approx(7.0, abs=0.01)
        assert severity == "HIGH"

    def test_control_four_measured_zeros_are_kept_as_measurements(self):
        """0.0 is a real answer to "is this group historically discriminated
        against" and must never be read as unsupplied: the event keeps all four
        and scores 0.5 (the drift boost alone), LOW, silently."""
        event, caught = _caught(
            FairnessAlertPrioritizer.build_drift_event,
            METRIC,
            ["B"],
            0.5,
            0.1,
            regulatory_risk=0.0,
            population_impact=0.0,
            drift_velocity=0.0,
            historical_discrimination=0.0,
        )
        (score, severity), warned = _caught(FairnessAlertPrioritizer().calculate_priority, event)

        assert caught == []
        assert warned == []
        assert event["regulatory_risk"] == 0.0
        assert score == pytest.approx(0.5)
        assert severity == "LOW"


# ===========================================================================
# CAUSE 6a. The Tier-3 view got neither of the fixes the other two tiers got.
# ===========================================================================


class TestTheTechnicalViewDisclosesWhatTheOtherTiersDisclose:
    def test_the_alert_panel_over_an_empty_store_is_not_painted_pass_green(self):
        """BEFORE: y=(0, 0, 0, 0) with marker_color=('#dc2626', '#f59e0b',
        '#059669', '#94a3b8') and the six subplot titles as the only
        annotations, while create_alert_summary on the identical store
        neutralised all four bars and said 'NOT AN ALL-CLEAR'."""
        fig = _quiet(FairnessDashboard(MetricsStore(), CFG).create_technical_view)
        bar = _severity_bar(fig)

        assert set(bar.marker.color) == {CFG.color_unknown}, bar.marker.color
        assert [t for t in _texts(fig) if "NOT AN ALL-CLEAR" in t], _texts(fig)

    def test_the_gauge_carries_the_drift_coverage_note_and_keeps_its_titles(self):
        """BEFORE: gauge value=100.0 on a store with ZERO drift rows and no
        occurrence of the word drift anywhere on the figure, while the gauge,
        the executive view and the operational view all carried 'Drift
        stability: NOT ASSESSED' for that same store. add_annotation, or the six
        subplot titles (which ARE layout annotations) are deleted."""
        store = _clean_store()
        assert len(_quiet(store.get_drift_history)) == 0

        fig = _quiet(FairnessDashboard(store, CFG).create_technical_view)
        texts = _texts(fig)

        assert [t.value for t in fig.data if t.type == "indicator"] == [100.0]
        assert [t for t in texts if "Drift stability: NOT ASSESSED" in t], texts
        assert "Health Score" in texts and "Drift Timeline" in texts, texts

    def test_the_drift_panel_colours_a_refused_window_neither_green_nor_red(self):
        """The (3,1) panel drew the scatter with NO marker colours at all, so a
        refused window and a measured clean one were the same dot."""
        store = _clean_store()
        store.ingest_drift_result(_DriftResult(score=0.05, detected=False, hours_ago=3))
        store.ingest_drift_result(_DriftResult(score=0.40, detected=None, hours_ago=2))

        fig = _quiet(FairnessDashboard(store, CFG).create_technical_view)
        scatter = [t for t in fig.data if t.name == "Drift"][0]

        assert list(scatter.marker.color) == [CFG.color_pass, CFG.color_unknown]

    # OVER-CORRECTION CONTROL

    def test_control_a_measured_store_keeps_its_verdict_palette_and_says_nothing(self):
        """Four determined records, a real alert determination on each and one
        drift row: the four severity bars keep the graded palette and no
        coverage note is added."""
        store = _clean_store()
        store.ingest_drift_result(_DriftResult())

        fig = _quiet(FairnessDashboard(store, CFG).create_technical_view)
        bar = _severity_bar(fig)

        assert tuple(bar.marker.color) == (
            CFG.color_fail,
            CFG.color_warn,
            CFG.color_pass,
            CFG.color_unknown,
        )
        assert not [t for t in _texts(fig) if "NOT AN ALL-CLEAR" in t]
        assert not [t for t in _texts(fig) if "NOT ASSESSED" in t], _texts(fig)


# ===========================================================================
# CAUSE 6b. The drift timeline's commonest unmeasured input said nothing.
# ===========================================================================


class TestTheDriftTimelineNamesAnUnrunDriftTest:
    def test_an_empty_drift_history_says_no_drift_test_was_run(self):
        """BEFORE: traces 0, annotations [], title 'Drift Score Timeline'. The
        two sibling charts in the same file already answered this shape of
        absence ('no trend was computed. This is not a flat trend.', 'NOT AN
        ALL-CLEAR')."""
        fig = _quiet(FairnessDashboard(MetricsStore(), CFG).create_drift_timeline)
        texts = _texts(fig)

        assert len(fig.data) == 0
        assert texts, "an empty drift timeline carried no statement at all"
        assert "No drift test was run" in texts[0]
        assert "not an absence of drift" in texts[0]

    def test_a_clean_verdict_over_an_unmeasured_score_is_not_painted_green(self):
        """BEFORE: one row with overall_drift_score=nan and drift_detected=False
        rendered y=[nan] colors=['#059669'] with annotations ['Alert threshold']:
        a PASS GREEN marker over a magnitude nobody measured, because n_unknown
        counted only 'd is None' and never looked at the score."""
        store = MetricsStore()
        store.ingest_drift_result(_DriftResult(score=NAN, detected=False))

        fig = _quiet(FairnessDashboard(store, CFG).create_drift_timeline)

        assert list(fig.data[0].marker.color) == [CFG.color_unknown]
        assert [t for t in _texts(fig) if "could not be checked" in t], _texts(fig)

    def test_a_refused_verdict_is_still_slate(self):
        """The READINESS-6 state this chart already had must survive the fix."""
        store = MetricsStore()
        store.ingest_drift_result(_DriftResult(score=0.4, detected=None))

        fig = _quiet(FairnessDashboard(store, CFG).create_drift_timeline)

        assert list(fig.data[0].marker.color) == [CFG.color_unknown]

    # OVER-CORRECTION CONTROL

    def test_control_a_fully_determined_timeline_carries_no_disclosure(self):
        """A measured clean window stays green, a measured drift stays red, and
        nothing on the figure says anything could not be checked."""
        store = MetricsStore()
        store.ingest_drift_result(_DriftResult(score=0.05, detected=False, hours_ago=3))
        store.ingest_drift_result(_DriftResult(score=0.70, detected=True, hours_ago=2))

        fig = _quiet(FairnessDashboard(store, CFG).create_drift_timeline)

        assert list(fig.data[0].marker.color) == [CFG.color_pass, CFG.color_fail]
        assert _texts(fig) == ["Alert threshold"], _texts(fig)


# ===========================================================================
# THE RE-AIMED SABOTAGES. _drift_coverage_note has two branches that both
# return the note, and the recorded sabotage of either one left 32 tests
# passing: for the fixture store the ABSENT-key branch answers first, so
# breaking the n_drift branch changed nothing, and breaking the absent-key
# branch let the n_drift branch answer instead. A sabotage that cannot fail
# looks exactly like a pin that holds. Each branch is now reached on its own.
# ===========================================================================


class TestEachCoverageNoteBranchIsPinnedOnItsOwn:
    def test_an_absent_drift_component_is_disclosed(self):
        """Branch 1, the one every MetricsStore reaches: the producer drops the
        drift_stability key when no drift test ran, so absence IS the signal."""
        note = _quiet(_drift_coverage_note, _clean_store(), _health({"metric_compliance": 100.0}))

        assert note is not None
        assert "NOT ASSESSED" in note
        assert "absent from this score" in note

    def test_a_present_drift_component_with_no_drift_row_is_disclosed(self):
        """Branch 2, which no store-computed HealthScore reaches today and which
        the recorded sabotage aimed at: a component claiming a drift stability
        over a drift table holding nothing. Reached directly, so breaking THIS
        branch alone reddens THIS test."""
        note = _quiet(_drift_coverage_note, MetricsStore(), _health({"drift_stability": 100.0}))

        assert note is not None
        assert "NOT ASSESSED" in note
        assert "20 percent drift weight" in note

    def test_a_coverage_question_that_itself_failed_says_could_not_check(self):
        """Branch 3, the third state: if the drift history cannot be READ, the
        answer is could-not-check, not 'assessed' and not 'not assessed'."""

        class _Unreadable:
            def get_drift_history(self, *a, **k):
                raise RuntimeError("drift table unavailable")

        note = _quiet(_drift_coverage_note, _Unreadable(), _health({"drift_stability": 100.0}))

        assert note is not None
        assert "COULD NOT CHECK" in note
        assert "drift table unavailable" in note

    # OVER-CORRECTION CONTROL

    def test_control_a_measured_drift_component_is_silent(self):
        """One drift row and a component computed from it: no note at all. A
        note that always fired would be the over-correction, and it is the
        mirror the auditor ran as S20."""
        store = _clean_store()
        store.ingest_drift_result(_DriftResult())

        note = _quiet(_drift_coverage_note, store, _health({"drift_stability": 96.0}))

        assert note is None
