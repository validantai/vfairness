"""BGL4 audit A-operations-1: the overturns, each as an executable demonstration.

WHAT THIS FILE WAS. An AUDITOR wrote it and changed nothing in the library. Every
test below asserts the behaviour the graded unit SHOULD have and was marked
``xfail(strict=True)``, so the suite stayed green while the defect was recorded,
and the moment somebody fixed one the test XPASSed, strict turned that into a
failure, and whoever fixed it removed the marker. A test that asserted the buggy
behaviour instead would go red on the fix, which is backwards.

WHAT THIS FILE IS NOW. BGL5 batch A-operations-1 closed all eleven overturned rows
on 2026-09-27, five root causes across ``operations/cicd/gate.py``,
``operations/monitoring/alerts.py`` and ``operations/reporting/dashboard.py``. All
twelve tests XPASSed on the fixed code, which is the evidence each fix reaches the
surface the auditor measured. Every marker is therefore removed and every test kept:
same name, same docstring, same subject, now asserting the corrected behaviour
rather than predicting it. The MEASURED-ON-HEAD block in each docstring is the
before; the assertion is the after. The refusal side of each fix, plus the
over-correction control that healthy input still gets its real number, is in
``tests/test_bgl5_operations_1.py``.

Each class names the grade it overturns and the batch's own claim about it.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    GateStatus,
    ModelFairnessGate,
)
from vfairness.operations.monitoring.alerts import FairnessAlertPrioritizer
from vfairness.operations.reporting.dashboard import DashboardConfig, FairnessDashboard
from vfairness.operations.reporting.store import MetricsStore

CFG = DashboardConfig()
METRIC = "demographic_parity_difference"


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _gate(**kw):
    cfg = GateConfig(metrics=[METRIC], thresholds={METRIC: 0.5}, **kw)
    return ModelFairnessGate(config=cfg)


def _fair_arrays(n: int = 100):
    y = np.array([1, 0] * (n // 2) * 2)
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


def _texts(fig):
    return [a.text or "" for a in (fig.layout.annotations or ())]


# ===========================================================================
# OVERTURN 1. build_drift_event, graded PROVEN ("it does not invent a drift
# verdict"). It invents FOUR CONTEXT FACTORS, and the unit's own docstring
# records that as "BGL g014, 2026-09-17, DEFECT NOT CLOSED, disclosed only".
# ===========================================================================


# BGL4 grade 12: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_build_drift_event_invents_four_context_factors():
    """MEASURED ON HEAD:

        build_drift_event("demographic_parity_difference", ["B"], 0.5, 0.1)
          -> {..., 'regulatory_risk': 0.5, 'population_impact': 0.3,
                   'drift_velocity': 0.3, 'historical_discrimination': 0.5}
        calculate_priority(that event)       -> (5.1, 'HIGH')   slack
        calculate_priority({"drift_score": 0.5, "mean_shift": 0.1}) -> (nan, 'UNSCORED')

    The same unmeasured context by two routes, one graded HIGH and one refused.
    The two tests named as evidence for this grade assert only that drift_score and
    mean_shift propagate as NaN, and both of them call this helper with the four
    context factors left out, so their own fixture carries the fabrication.
    """
    event = _quiet(FairnessAlertPrioritizer.build_drift_event, METRIC, ["B"], 0.5, 0.1)
    for factor in (
        "regulatory_risk",
        "population_impact",
        "drift_velocity",
        "historical_discrimination",
    ):
        assert factor not in event or event[factor] is None, (
            f"{factor} was not supplied by the caller and nobody measured it, yet the "
            f"event carries {event.get(factor)!r}, which calculate_priority cannot tell "
            f"from a measurement"
        )


# BGL4 grade 12: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_an_alert_built_from_substituted_context_is_graded_high():
    """The substitution is not merely present, it decides the severity and the route,
    and the message a person reads off the channel says nothing about it."""
    prioritizer = FairnessAlertPrioritizer()
    event = _quiet(FairnessAlertPrioritizer.build_drift_event, METRIC, ["B"], 0.5, 0.1)
    alert = _quiet(prioritizer.create_alert, event)

    assert alert.severity == prioritizer.UNSCORED_SEVERITY, (
        f"an alert whose four context factors were all substituted was graded "
        f"{alert.severity!r} at priority {alert.priority_score} and routed to "
        f"{alert.routing}; the same context supplied honestly scores UNSCORED"
    )


# ===========================================================================
# OVERTURN 2. evaluate and evaluate_from_metrics, both graded PROVEN. A NaN
# BASELINE turns require_improvement AND allow_degradation_margin into no-ops
# and the gate reports "All fairness requirements met".
# ===========================================================================


# BGL4 grade 9: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_evaluate_approves_when_the_improvement_requirement_could_not_be_checked():
    """MEASURED ON HEAD, require_improvement=True, improvement_margin=0.01, baseline NaN:

        approved=True  status=GateStatus.APPROVED
        summary 'APPROVED - All fairness requirements met'
        metric row {'value': 0.0, 'passed': True, 'baseline_value': nan,
                    'improvement': nan, 'message': ''}
        markdown '| demographic_parity_difference | 0.0000 | 0.5000 | (tick) Pass |'
        create_github_check -> conclusion 'success'

    ``improvement < margin`` is False for NaN, so the configured requirement was
    never evaluated. The metric VALUE is guarded against NaN three ways in this
    method; the BASELINE it is compared against is not guarded at all.
    """
    gate = _gate(require_improvement=True, improvement_margin=0.01)
    y_true, y_pred, protected = _fair_arrays()

    decision = gate.evaluate(y_true, y_pred, protected, baseline_metrics={METRIC: float("nan")})

    assert decision.approved is False or decision.warnings, (
        f"the gate reported {decision.summary!r} with warnings={decision.warnings} for a "
        f"required improvement it could not compute (improvement="
        f"{decision.metric_evaluations[0].improvement})"
    )
    # The corrected behaviour, tightened now that it is a pin rather than a
    # prediction: fail closed with the reason named, and the check not green.
    assert decision.approved is False
    assert decision.status is GateStatus.BLOCKED
    assert any("baseline" in r for r in decision.blocking_reasons), decision.blocking_reasons
    assert gate.create_github_check(decision)["conclusion"] == "failure"


# BGL4 grade 10: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_evaluate_from_metrics_approves_an_uncheckable_improvement():
    """The same hole on the entry point a CI system hands pre-computed numbers to."""
    gate = _gate(require_improvement=True, improvement_margin=0.01)

    decision = gate.evaluate_from_metrics({METRIC: 0.02}, baseline_metrics={METRIC: float("nan")})

    assert decision.approved is False or decision.warnings, (
        f"{decision.summary!r} over an improvement of {decision.metric_evaluations[0].improvement}"
    )
    assert decision.approved is False
    assert decision.status is GateStatus.BLOCKED
    assert any("baseline" in r for r in decision.blocking_reasons), decision.blocking_reasons


# ===========================================================================
# OVERTURN 3. evaluate_hierarchical, graded PROVEN, and its judgement quotes a
# "blocking reason" as observed evidence. IntersectionalGateDecision has no
# field for one, so the reason the method builds is DISCARDED.
# ===========================================================================


# BGL4 grade 11: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_hierarchical_refusal_reason_reaches_no_consumer():
    """MEASURED ON HEAD, evaluate_hierarchical(y, y, {}):

        summary            'BLOCKED - 0/0 levels passed'
        to_dict()          {'approved': false, 'status': 'blocked', 'level_results': {},
                            'small_sample_warnings': [], 'summary_decision': null, ...}
        to_markdown_report '# Hierarchical Fairness Gate Report / **Status**: BLOCKED /
                            **Levels evaluated**: 0'
        report card        '**Result**: Deployment blocked'

    The sentence the method appends to its local ``all_blocking`` list, "no level was
    evaluated, so nothing was checked; the gate fails closed rather than approving an
    unevaluated deployment", is on none of those surfaces: the dataclass declares no
    blocking_reasons field and the list is dropped at the constructor. A reader gets
    BLOCKED, which is the FAIL state, for a could-not-check.
    """
    gate = _gate()
    y_true, y_pred, _ = _fair_arrays()

    decision = gate.evaluate_hierarchical(y_true, y_pred, {})
    surfaces = (
        json.dumps(decision.to_dict(), default=str)
        + decision.to_markdown_report()
        + FairnessReportCard(decision).to_markdown()
        + decision.summary
    )

    assert "nothing was checked" in surfaces, (
        "the refusal reason is on no consumer surface; the whole decision reads as a "
        "measured deployment block"
    )


# ===========================================================================
# OVERTURN 4. FairnessReportCard.to_markdown (and the to_dict and the GitHub
# payload that carry it), all graded PROVEN, plus
# IntersectionalGateDecision.to_markdown_report, graded SEMI-PROVEN. The
# HIERARCHICAL branch renders a never-computed metric as a FAIL with no
# statement anywhere that it was not computed, which is verbatim the property
# the named test asserts for the FLAT branch. Coverage of the named test file
# over _format_hierarchical: 0 of 22 body lines.
# ===========================================================================


def _unmeasurable_hierarchical_decision():
    """One group in the attribute, so no disparity is definable at any level."""
    y_true, y_pred, _ = _fair_arrays()
    return _gate().evaluate_hierarchical(y_true, y_pred, {"gender": np.array(["a"] * 200)})


# BGL4 grades 0, 1, 2: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_hierarchical_report_card_shows_a_fail_it_never_measured():
    """MEASURED ON HEAD:

        #### overall: Failed Metrics
        - **demographic_parity_difference**: nan (threshold: 0.5000)

    and the strings 'could not be computed', 'fails closed', 'not measured' and
    'could-not-check' are all ABSENT from the whole card. The flat path carries the
    reason because _format_simple prints decision.blocking_reasons; _format_hierarchical
    prints no reasons section at all.
    """
    markdown = FairnessReportCard(
        _unmeasurable_hierarchical_decision(), model_name="loan-v3"
    ).to_markdown()

    assert "could not be computed" in markdown, (
        "the PR comment showed a FAIL for a metric that was never computed, with no "
        "statement anywhere that it was not computed"
    )


# BGL4 grades 0, 1: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_card_dict_and_github_payload_carry_the_same_silence():
    """to_dict()['markdown'] and to_github_comment_payload()['body'] are that string,
    so the defect is on every one of the three graded serialisers at once."""
    card = FairnessReportCard(_unmeasurable_hierarchical_decision(), model_name="m")

    assert "could not be computed" in card.to_dict()["markdown"]
    assert "could not be computed" in card.to_github_comment_payload()["body"]


# BGL4 grade 6: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_hierarchical_markdown_report_prints_fail_for_an_unmeasured_metric():
    """IntersectionalGateDecision.to_markdown_report has no Blocking Issues and no
    Warnings section, so 'nan | 0.5000 | Fail' is the whole story a reader gets."""
    report = _unmeasurable_hierarchical_decision().to_markdown_report()

    assert "could not be computed" in report or "not measured" in report, report


# ===========================================================================
# OVERTURN 5. create_technical_view, graded PROVEN. The Tier-3 view is the third
# rendering of two quantities this very batch fixed on the other two, and it got
# neither fix.
# ===========================================================================


def _clean_store():
    return _store(
        ["gender_male", "gender_female", "gender_male", "gender_female"],
        [0.02, 0.03, 0.01, 0.04],
    )


# BGL4 grade 15: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_technical_view_alert_panel_is_painted_pass_green_over_nothing():
    """MEASURED ON HEAD, create_technical_view() on an EMPTY MetricsStore:

        alert bar y=(0, 0, 0, 0)
        marker_color=('#dc2626', '#f59e0b', '#059669', '#94a3b8')
        annotations = the six subplot titles only

    create_alert_summary() on the identical store neutralises all four bars to
    '#94a3b8' and carries 'NOT AN ALL-CLEAR: no metric record in this window was
    compared to a threshold'. Same store, same wave, two different answers.
    """
    fig = _quiet(FairnessDashboard(MetricsStore(), CFG).create_technical_view)
    sev_bar = [t for t in fig.data if t.type == "bar" and "LOW" in [str(x) for x in t.x]][0]

    assert set(sev_bar.marker.color) == {CFG.color_unknown}, (
        f"a verdict palette {sev_bar.marker.color} over a store that was never given "
        f"anything; PASS green is {CFG.color_pass}"
    )


# BGL4 grade 15: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_the_technical_view_gauge_omits_the_drift_coverage_note():
    """MEASURED ON HEAD, four determined clean records and ZERO drift rows (the default
    state of every MetricsStore): the Tier-3 gauge renders value=100.0 and no
    annotation mentions drift, while create_health_score_gauge,
    create_executive_view and create_operational_view all carry 'Drift stability: NOT
    ASSESSED' on the identical store, each pinned by a test in this same batch."""
    fig = _quiet(FairnessDashboard(_clean_store(), CFG).create_technical_view)

    assert [t for t in _texts(fig) if "NOT ASSESSED" in t], _texts(fig)


# ===========================================================================
# OVERTURN 6. GateDecision.to_dict, graded SEMI-PROVEN. The uncovered input
# class collapses a documented third state.
# ===========================================================================


# BGL4 grade 3: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_an_unchecked_small_sample_list_serialises_as_none_found():
    """evaluate_from_metrics' own docstring: "small_sample_warnings on the returned
    decision is always empty here, and an empty list means NOT CHECKED, not 'no small
    groups'". to_dict() emits [] for both, with no key naming which one it is, so the
    distinction the docstring insists on does not reach a consumer at all."""
    gate = _gate()
    y_true, y_pred, protected = _fair_arrays()

    checked = gate.evaluate(y_true, y_pred, protected).to_dict()
    never_checked = gate.evaluate_from_metrics({METRIC: 0.02}).to_dict()

    assert checked["small_sample_warnings"] != never_checked["small_sample_warnings"] or any(
        "sample" in k and k != "small_sample_warnings" for k in never_checked
    ), (
        "the checked-and-clean decision and the never-checked one serialise "
        f"byte-identically as {never_checked['small_sample_warnings']!r}"
    )


# ===========================================================================
# OVERTURN 7. create_drift_timeline, graded PROVEN on drift_detected=None. The
# commonest unchecked state, NO DRIFT ROW AT ALL, is drawn with nothing on the
# figure, although the two sibling charts in this same file got exactly that
# sentence in this same wave.
# ===========================================================================


# BGL4 grade 25: XPASSED once BGL5 A-operations-1 closed it on 2026-09-27,
# so the xfail(strict=True) marker is gone and this now pins the fix.
def test_overturn_an_empty_drift_history_says_nothing_at_all():
    """MEASURED ON HEAD: 0 traces, 0 annotations, title 'Drift Score Timeline'.
    create_trend_analysis says 'no trend was computed. This is not a flat trend.' and
    create_alert_summary says 'NOT AN ALL-CLEAR' for the same shape of absence."""
    fig = _quiet(FairnessDashboard(MetricsStore(), CFG).create_drift_timeline)

    assert len(fig.data) == 0
    # Measured while sabotaging this pin: an annotation whose TEXT is empty still
    # makes _texts(fig) a non-empty list, so "there is an annotation" does not
    # discriminate. The sentence itself has to be asserted.
    assert "No drift test was run" in " ".join(_texts(fig)), _texts(fig)
    assert _texts(fig), (
        "an empty drift timeline under the title 'Drift Score Timeline' carries no "
        "statement that no drift test was run"
    )
