"""BGL3 operations-1: the reporting surfaces published clean-looking values.

Fourteen units across ``operations/reporting/dashboard.py`` and
``operations/reporting/reports.py``, executed on healthy input and on input where
the quantity they show genuinely does not exist. Measured on the real figures and
the real rendered documents, because a colour and a sentence are the only things
a reader gets.

What was measured before the fixes, in numbers:

* ``create_health_score_gauge`` on a store with four clean determined metric
  records and ZERO drift rows (the default state of every MetricsStore) rendered
  ``value=100.0`` captioned "Fairness health score is 100/100 (healthy) with a
  stable trend", and the word drift appeared NOWHERE on the figure, although 20
  percent of that 100 is ``drift_stability=100.0``, a store default for an empty
  drift table. ``ReportGenerator._health_with_drift_coverage`` already refuses to
  publish that unqualified on the document surface.
* ``create_executive_view`` on an EMPTY store rendered the Active Alerts KPI as
  ``value=0`` in ``#059669`` PASS GREEN with the title "Active Alerts", byte for
  byte identical to a store holding two measured clean records. The READINESS-6
  guard keyed only on ``n_not_assessable``, which the store leaves at 0 on the
  path where nothing was ingested at all.
* the same view rendered Metric Compliance as ``value=0`` with
  ``delta={'reference': 80}`` for ``components={}``, beside a health gauge that
  correctly rendered ``value=None``.
* ``create_alert_summary`` rendered ``y=(0, 0, 0)`` with
  ``marker_color=('#dc2626', '#f59e0b', '#059669')`` and no annotation for an
  EMPTY store AND for a store with four measured clean records, while the store
  itself warned on that very call that "the empty result means nothing has been
  ingested. It is NOT a finding that no alert fired."
* the same chart counted an alert carrying no severity as LOW: ``y=(1, 0, 1)``
  with the unscored alert in the green LOW bar, while the store's own record for
  it carried ``metadata={'severity': 'UNSCORED'}``.
* ``create_trend_analysis`` asked for ``['gender_female', 'gender_nonbinary']``
  over a store holding no nonbinary record drew ONE trace and no annotation.
* ``create_intersectional_heatmap('...', 'gender', 'age')`` over a store with no
  age group rendered ``z=([0.9], [0.02])`` on the RdYlGn ramp under the title
  "Intersectional Analysis" with no note that no intersection was built.
* ``generate_threshold_breach_report(metric, float('nan'), 0.10)`` published
  "Current Value: nan", "Breach Magnitude: nan% beyond threshold" and the
  recommendation "The demographic_parity_difference metric has breached its 0.10
  threshold", over a comparison that cannot happen: NaN compares False in both
  directions.
* ``generate_alert_report`` printed "Affected Groups: []" under "Severity:
  CRITICAL".
* ``_render_html`` swallowed a failing figure into ``charts_html = ""``, so a
  crashed chart and a report configured without charts produced the same page.

Every control test here exists because a unit that refuses everything is as
wrong as one that answers everything.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting.dashboard import DashboardConfig, FairnessDashboard
from vfairness.operations.reporting.reports import OutputFormat, ReportGenerator
from vfairness.operations.reporting.store import MetricsStore

CFG = DashboardConfig()
METRIC = "demographic_parity_difference"


# ===== fixtures


def _store(groups, values, alerts=None, determined=None):
    """A store holding one metric over *groups*, with explicit determinations."""
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
                "alert": alerts if alerts is not None else [False] * n,
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
    """Four measured, threshold-compared, non-breaching records."""
    return _store(
        ["gender_male", "gender_female", "gender_male", "gender_female"],
        [0.02, 0.03, 0.01, 0.04],
    )


class _DriftResult:
    """The shape MetricsStore.ingest_drift_result reads."""

    def __init__(self, score=0.05, detected=False):
        self.timestamp = datetime.now() - timedelta(hours=1)
        self.overall_drift_score = score
        self.drift_detected = detected
        self.metric = METRIC
        self.mmd_score = 0.0
        self.worst_scale = None


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn()


def _indicator(fig, title_startswith):
    for trace in fig.data:
        if getattr(trace, "type", "") == "indicator" and (trace.title.text or "").startswith(
            title_startswith
        ):
            return trace
    raise AssertionError(f"no indicator titled {title_startswith!r}")


def _texts(fig):
    return [a.text or "" for a in (fig.layout.annotations or ())]


# ===== the gauge and the unmeasured 20 percent


def test_a_gauge_over_no_drift_row_does_not_publish_an_unqualified_100():
    """BEFORE: value=100.0 with the single annotation "Fairness health score is
    100/100 (healthy) with a stable trend", and no occurrence of the word drift,
    on a store with 0 drift rows whose drift_stability component is the store's
    100.0 default for an empty table."""
    store = _clean_store()
    assert len(_quiet(store.get_drift_history)) == 0, "the fixture must hold no drift row"
    health = _quiet(store.compute_health_score)
    # ABSENT, not 100.0. This asserted components["drift_stability"] == 100.0 as the
    # PRECONDITION: the fabricated default had to be present for the gauge's silence
    # about it to be the defect. The producer was fixed later the same day and now drops
    # the key and renormalises, so asserting the 100.0 would be asserting the
    # fabrication. The stronger statement is that it is gone, and the gauge must STILL
    # carry the disclosure, which is what the rest of this test checks.
    assert "drift_stability" not in health.components, health.components

    fig = _quiet(FairnessDashboard(store, CFG).create_health_score_gauge)

    assert fig.data[0].value == pytest.approx(100.0), "the measured score must still be shown"
    disclosure = [t for t in _texts(fig) if "Drift stability: NOT ASSESSED" in t]
    assert disclosure, (
        f"a 100/100 gauge carrying an unmeasured 20 percent said nothing about it; "
        f"annotations were {_texts(fig)}"
    )
    assert "not a drift stability of 100" in disclosure[0]


def test_control_a_gauge_with_a_drift_row_carries_no_coverage_note():
    """OVER-CORRECTION CONTROL. A store that DID run a drift test must not be
    told its drift component is unmeasured."""
    store = _clean_store()
    store.ingest_drift_result(_DriftResult())

    fig = _quiet(FairnessDashboard(store, CFG).create_health_score_gauge)

    assert not [t for t in _texts(fig) if "NOT ASSESSED" in t], _texts(fig)


def test_the_executive_view_carries_the_same_coverage_note_as_the_gauge():
    """The Tier-1 composite is drawn twice and must not disclose differently."""
    fig = _quiet(FairnessDashboard(_clean_store(), CFG).create_executive_view)

    assert [t for t in _texts(fig) if "Drift stability: NOT ASSESSED" in t], _texts(fig)


def test_the_operational_view_carries_the_same_coverage_note_and_keeps_its_titles():
    """add_annotation, not update_layout(annotations=), or the four subplot
    titles of this grid are deleted by the disclosure."""
    fig = _quiet(FairnessDashboard(_clean_store(), CFG).create_operational_view)
    texts = _texts(fig)

    assert [t for t in texts if "Drift stability: NOT ASSESSED" in t], texts
    assert "Health Score" in texts and "Group Disparity" in texts, texts


# ===== the executive KPI cards


def test_the_alert_kpi_over_an_empty_store_is_not_painted_pass_green():
    """BEFORE: an EMPTY store and a store with two measured clean records BOTH
    rendered the Active Alerts KPI as value=0 with number.font.color '#059669'
    and the title 'Active Alerts'. The empty store's health score is
    score=None / status='not_assessed' / n_metrics=0, but n_not_assessable=0, so
    the READINESS-6 guard that keyed only on that count never fired."""
    health = _quiet(MetricsStore().compute_health_score)
    assert health.n_not_assessable == 0 and health.score is None, health

    kpi = _indicator(
        _quiet(FairnessDashboard(MetricsStore(), CFG).create_executive_view), "Active Alerts"
    )

    assert kpi.number.font.color == CFG.color_unknown, (
        f"zero alerts over a store that was never given anything was painted "
        f"{kpi.number.font.color}; PASS green is {CFG.color_pass}"
    )
    assert "no metric record was assessed" in (kpi.title.text or "")


def test_control_the_alert_kpi_over_a_measured_clean_window_is_still_green():
    """OVER-CORRECTION CONTROL. A measured, determined, alert-free window must
    still be allowed to read clean."""
    kpi = _indicator(
        _quiet(FairnessDashboard(_clean_store(), CFG).create_executive_view), "Active Alerts"
    )

    assert kpi.number.font.color == CFG.color_pass
    assert (kpi.title.text or "") == "Active Alerts"


def test_an_absent_metric_compliance_is_not_rendered_as_zero():
    """BEFORE: value=0 with delta={'reference': 80} for components={}, i.e. a
    measured total non-compliance and a -80 delta for a quantity nobody
    computed, beside a health gauge that correctly rendered value=None."""
    kpi = _indicator(
        _quiet(FairnessDashboard(MetricsStore(), CFG).create_executive_view), "Metric Compliance"
    )

    assert kpi.value != 0, "an indicator reading 0 is a measurement"
    assert np.isnan(kpi.value)
    assert kpi.mode == "number", "a delta asserts a comparison that was never made"
    assert kpi.number.font.color == CFG.color_unknown
    assert "not assessed" in (kpi.number.prefix or "")


def test_control_a_measured_metric_compliance_keeps_its_number_and_delta():
    """OVER-CORRECTION CONTROL."""
    kpi = _indicator(
        _quiet(FairnessDashboard(_clean_store(), CFG).create_executive_view), "Metric Compliance"
    )

    assert kpi.value == pytest.approx(100.0)
    assert kpi.delta.reference == 80


# ===== the alert summary


def test_an_alert_with_no_severity_is_not_counted_as_low():
    """BEFORE: y=(1, 0, 1) for one dict-shaped alert with no severity key plus
    one CRITICAL, i.e. the unscored alert counted in the LOW bar and painted
    '#059669', while MetricsStore.ingest_alert recorded its own copy of that
    alert with metadata={'severity': 'UNSCORED'}."""
    store = MetricsStore()
    _quiet(
        lambda: store.ingest_alert(
            {"metric_name": METRIC, "message": "who knows", "timestamp": datetime.now()}
        )
    )
    _quiet(
        lambda: store.ingest_alert(
            {
                "metric_name": METRIC,
                "message": "bad",
                "severity": "CRITICAL",
                "priority_score": 9.1,
                "timestamp": datetime.now(),
            }
        )
    )

    bar = _quiet(FairnessDashboard(store, CFG).create_alert_summary).data[0]
    counts = dict(zip([str(x) for x in bar.x], list(bar.y)))
    colors = dict(zip([str(x) for x in bar.x], list(bar.marker.color)))

    assert counts["LOW"] == 0, f"an unscored alert was counted as LOW: {counts}"
    assert counts["UNSCORED"] == 1, counts
    assert counts["CRITICAL"] == 1, counts
    assert colors["UNSCORED"] == CFG.color_unknown, (
        f"the unscored bar was painted {colors['UNSCORED']}; PASS green is {CFG.color_pass}"
    )
    assert len(bar.marker.color) == len(bar.x), "one colour per bar, or plotly picks its own"


def test_control_a_real_low_alert_is_still_counted_as_low_and_green():
    """OVER-CORRECTION CONTROL. A severity that WAS determined keeps its bucket."""
    store = MetricsStore()
    _quiet(
        lambda: store.ingest_alert(
            {
                "metric_name": METRIC,
                "message": "minor",
                "severity": "LOW",
                "priority_score": 1.5,
                "timestamp": datetime.now(),
            }
        )
    )

    bar = _quiet(FairnessDashboard(store, CFG).create_alert_summary).data[0]
    counts = dict(zip([str(x) for x in bar.x], list(bar.y)))
    colors = dict(zip([str(x) for x in bar.x], list(bar.marker.color)))

    assert counts["LOW"] == 1 and counts["UNSCORED"] == 0, counts
    assert colors["LOW"] == CFG.color_pass


def test_zero_alert_counts_over_a_store_that_examined_nothing_are_not_an_all_clear():
    """BEFORE: an EMPTY store and a store with four measured clean records both
    rendered y=(0, 0, 0), marker_color=('#dc2626', '#f59e0b', '#059669') and
    zero annotations, while store.get_alerts warned on the same call that the
    empty result "is NOT a finding that no alert fired"."""
    fig = _quiet(FairnessDashboard(MetricsStore(), CFG).create_alert_summary)
    bar = fig.data[0]

    assert set(bar.y) == {0}
    assert [t for t in _texts(fig) if "NOT AN ALL-CLEAR" in t], _texts(fig)
    assert set(bar.marker.color) == {CFG.color_unknown}, (
        f"a verdict palette was drawn over a window nobody examined: {bar.marker.color}"
    )


def test_control_zero_alerts_over_measured_records_carries_no_disclosure():
    """OVER-CORRECTION CONTROL. Four determined, non-breaching records ARE
    evidence, so this chart must be allowed to show three plain zeros."""
    fig = _quiet(FairnessDashboard(_clean_store(), CFG).create_alert_summary)
    bar = fig.data[0]

    assert _texts(fig) == [], _texts(fig)
    assert list(bar.marker.color)[:3] == [CFG.color_fail, CFG.color_warn, CFG.color_pass]


def test_undetermined_records_are_named_beside_the_alert_counts():
    """A record nobody compared to a threshold cannot raise an alert, so it
    cannot support a zero either. BEFORE: no annotation at all."""
    store = _store(
        ["gender_male", "gender_female"],
        [0.02, 0.9],
        alerts=[False, False],
        determined=[True, False],
    )

    fig = _quiet(FairnessDashboard(store, CFG).create_alert_summary)

    assert [t for t in _texts(fig) if "never compared to a threshold" in t], _texts(fig)
    assert [t for t in _texts(fig) if "1 of 2" in t], _texts(fig)


# ===== trend analysis


def test_a_requested_group_with_no_record_is_named_on_the_chart():
    """BEFORE: asking for ['gender_female', 'gender_nonbinary'] over a store
    holding gender_female and gender_male drew ONE trace, gender_female, with no
    annotation anywhere on the figure: a two-group comparison rendered as a
    one-group chart that looked complete."""
    store = _store(["gender_male", "gender_female"], [0.02, 0.9])

    fig = _quiet(
        lambda: FairnessDashboard(store, CFG).create_trend_analysis(
            METRIC, groups=["gender_female", "gender_nonbinary"]
        )
    )

    assert len(fig.data) == 1
    disclosure = [t for t in _texts(fig) if "NOT SHOWN" in t]
    assert disclosure, f"a dropped group was not named: {_texts(fig)}"
    assert "gender_nonbinary" in disclosure[0]


def test_control_two_present_groups_draw_two_traces_and_no_disclosure():
    """OVER-CORRECTION CONTROL."""
    store = _store(["gender_male", "gender_female"], [0.02, 0.9])

    fig = _quiet(
        lambda: FairnessDashboard(store, CFG).create_trend_analysis(
            METRIC, groups=["gender_female", "gender_male"]
        )
    )

    assert len(fig.data) == 2
    assert _texts(fig) == []


def test_an_empty_trend_window_says_it_computed_no_trend():
    """BEFORE: an empty store produced a figure with 0 traces, 0 annotations and
    a title reading "Trend Analysis: Demographic Parity Difference"."""
    fig = _quiet(lambda: FairnessDashboard(MetricsStore(), CFG).create_trend_analysis(METRIC))

    assert len(fig.data) == 0
    assert [t for t in _texts(fig) if "not a flat trend" in t], _texts(fig)


# ===== intersectional heatmap


def test_a_heatmap_that_could_not_build_an_intersection_says_so():
    """BEFORE: ("gender", "age") over a store with no age group rendered
    z=([0.9], [0.02]) on the RdYlGn ramp, x=('demographic_parity_difference',),
    yaxis 'gender', xaxis 'age', title "Intersectional Analysis: Demographic
    Parity Difference", and NO annotation: marginal per-group values presented
    as an intersectional analysis, in verdict colours."""
    store = _store(["gender_male", "gender_female"], [0.02, 0.9])

    fig = _quiet(
        lambda: FairnessDashboard(store, CFG).create_intersectional_heatmap(METRIC, "gender", "age")
    )

    disclosure = [t for t in _texts(fig) if "NOT AN INTERSECTIONAL ANALYSIS" in t]
    assert disclosure, f"a substituted chart claimed to be the requested one: {_texts(fig)}"
    assert "age" in disclosure[0] and "MARGINAL" in disclosure[0]


def test_a_grid_with_no_measured_intersection_reports_its_coverage():
    """BEFORE: four marginal groups and no intersection produced a 2x2 grid with
    0 of 4 cells populated and nothing on the figure saying so."""
    store = _store(
        ["gender_male", "gender_female", "race_black", "race_white"],
        [0.02, 0.9, 0.3, 0.05],
    )

    fig = _quiet(
        lambda: FairnessDashboard(store, CFG).create_intersectional_heatmap(
            METRIC, "gender", "race"
        )
    )
    z = np.asarray(fig.data[0].z, dtype=float)

    assert int(np.count_nonzero(~np.isnan(z))) == 0
    assert [t for t in _texts(fig) if "4 of 4 intersection(s) have no record" in t], _texts(fig)


def test_control_a_fully_measured_grid_carries_no_coverage_note():
    """OVER-CORRECTION CONTROL. Four ingested intersections beside the four
    levels they are made of: four populated cells, nothing to disclose.

    The levels are part of the fixture on purpose. The axis search reads levels,
    not composites, so a store holding ONLY intersections has no axes to build
    and takes the marginal fallback: that is what the fallback wording had to be
    corrected for while writing this control."""
    store = _store(
        [
            "gender_male_race_black",
            "gender_male_race_white",
            "gender_female_race_black",
            "gender_female_race_white",
            "gender_male",
            "gender_female",
            "race_black",
            "race_white",
        ],
        [0.31, 0.04, 0.44, 0.07, 0.02, 0.5, 0.3, 0.05],
    )

    fig = _quiet(
        lambda: FairnessDashboard(store, CFG).create_intersectional_heatmap(
            METRIC, "gender", "race"
        )
    )
    z = np.asarray(fig.data[0].z, dtype=float)

    assert int(np.count_nonzero(~np.isnan(z))) == 4
    assert _texts(fig) == [], _texts(fig)


# ===== refusals that already held


def test_refusal_a_group_whose_values_are_all_nan_is_not_coloured_as_a_verdict():
    """VERIFIED CORRECT, pinned so it stays correct. A group whose every value is
    NaN is painted the could-not-check slate '#94a3b8', not the fail red that
    `nan >= 0.8 -> False` used to produce."""
    store = _store(
        ["gender_male", "gender_male", "gender_female", "gender_female"],
        [np.nan, np.nan, 0.5, 0.4],
        alerts=[False, False, True, True],
        determined=[False, False, True, True],
    )

    bar = _quiet(lambda: FairnessDashboard(store, CFG).create_disparity_comparison(METRIC)).data[0]
    colors = dict(zip([str(g) for g in bar.x], list(bar.marker.color)))

    assert colors["gender_male"] == CFG.color_unknown, colors
    assert colors["gender_female"] == CFG.color_fail, colors


def test_refusal_the_drift_timeline_paints_an_unchecked_window_neither_green_nor_red():
    """VERIFIED CORRECT, pinned so it stays correct (READINESS-6 fixed it)."""
    store = MetricsStore()
    for score, detected in [(0.05, False), (0.42, True), (0.42, None)]:
        store.ingest_drift_result(_DriftResult(score, detected))

    fig = _quiet(FairnessDashboard(store, CFG).create_drift_timeline)
    colors = list(fig.data[0].marker.color)

    assert colors == [CFG.color_pass, CFG.color_fail, CFG.color_unknown], colors
    assert [t for t in _texts(fig) if "could not be checked" in t], _texts(fig)


def test_refusal_the_executive_report_over_an_empty_store_withholds_the_score():
    """VERIFIED CORRECT (C-04 / READINESS-5), pinned because four other units in
    this batch render the same HealthScore."""
    content = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_executive_report(OutputFormat.MARKDOWN)
    ).content

    assert "## Health Score: not assessed (NOT_ASSESSED)" in content
    assert "| Metric Compliance | not assessed |" in content
    assert "could NOT be assessed" in content
    assert "/100" not in content.split("## Summary")[0].replace("not assessed", "")


def test_refusal_an_all_undetermined_store_is_not_reported_as_compliant():
    """VERIFIED CORRECT (R-2 / H-10). Two NaN records nobody compared: the
    operational and technical reports name them rather than average them in."""
    store = _store(
        ["gender_male", "gender_female"],
        [np.nan, np.nan],
        alerts=[False, False],
        determined=[False, False],
    )
    gen = ReportGenerator(store)

    for content in (
        _quiet(lambda: gen.generate_operational_report(OutputFormat.MARKDOWN)).content,
        _quiet(lambda: gen.generate_technical_report(OutputFormat.MARKDOWN)).content,
    ):
        assert "COULD NOT CHECK: 2 of 2" in content, content
        assert "| Metric Compliance | not assessed |" in content, content


def test_refusal_a_drift_result_with_no_analysed_scale_is_not_called_stable():
    """VERIFIED CORRECT (H-11, reader side)."""

    class _NoScale:
        metric = METRIC
        overall_drift_score = float("nan")
        drift_detected = None
        mmd_score = None
        worst_scale = None

    rep = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_drift_report(
            _NoScale(), OutputFormat.MARKDOWN
        )
    )

    assert "Overall Drift Score: not measured" in rep.content
    assert "COULD NOT CHECK (no scale was analysed" in rep.content
    assert not [r for r in rep.recommendations if "No significant drift" in r]


# ===== threshold breach report


def test_a_breach_report_over_an_unmeasurable_value_does_not_publish_a_breach():
    """BEFORE: generate_threshold_breach_report(metric, float('nan'), 0.10)
    published "Current Value: nan", "Breach Magnitude: nan% beyond threshold" and
    the recommendation "The demographic_parity_difference metric has breached its
    0.10 threshold", over a comparison that cannot happen."""
    gen = ReportGenerator(_clean_store())

    rep = _quiet(
        lambda: gen.generate_threshold_breach_report(
            METRIC, float("nan"), 0.10, OutputFormat.MARKDOWN
        )
    )
    breach = rep.sections[0]["content"]

    assert "nan% beyond threshold" not in breach, breach
    assert "Current Value: not measured" in breach, breach
    assert "COULD NOT CHECK" in breach, breach
    assert not [r for r in rep.recommendations if "has breached" in r], rep.recommendations
    assert [r for r in rep.recommendations if "COULD NOT CHECK" in r], rep.recommendations

    payload = json.loads(
        _quiet(
            lambda: gen.generate_threshold_breach_report(
                METRIC, float("nan"), 0.10, OutputFormat.JSON
            )
        ).content.replace("NaN", "null")
    )
    assert payload["comparison_made"] is False
    assert payload["breach_pct"] is None


def test_control_a_real_breach_still_states_its_magnitude_and_asserts_the_breach():
    """OVER-CORRECTION CONTROL. 0.45 against 0.10 is 350.0 percent beyond it."""
    gen = ReportGenerator(_clean_store())

    rep = _quiet(
        lambda: gen.generate_threshold_breach_report(METRIC, 0.45, 0.10, OutputFormat.MARKDOWN)
    )

    assert "Breach Magnitude: 350.0% beyond threshold" in rep.sections[0]["content"]
    assert [r for r in rep.recommendations if "has breached its 0.10 threshold" in r]
    payload = json.loads(
        _quiet(
            lambda: gen.generate_threshold_breach_report(METRIC, 0.45, 0.10, OutputFormat.JSON)
        ).content
    )
    assert payload["comparison_made"] is True


def test_control_a_zero_threshold_keeps_the_wording_r2_gave_it():
    """A threshold of 0.0 is FINITE: the comparison is real and only the
    percentage beyond it is undefined. The new could-not-check branch must not
    swallow that case (tests/test_readiness_reports.py pins the wording)."""
    rep = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_threshold_breach_report(
            METRIC, 0.35, 0.0, OutputFormat.MARKDOWN
        )
    )
    breach = rep.sections[0]["content"]

    assert "not measurable (the threshold is 0" in breach, breach
    assert "COULD NOT CHECK" not in breach, breach
    assert [r for r in rep.recommendations if "has breached" in r]


# ===== alert report


def test_an_alert_that_records_no_group_does_not_print_an_empty_list():
    """BEFORE: "Affected Groups: []" printed under "Severity: CRITICAL". The
    producer builds that list as `affected_groups or
    drift_event.get("affected_groups", [])`, so [] is "nobody recorded who is
    affected", which reads as "nobody is affected"."""
    rep = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_alert_report(
            {"severity": "CRITICAL", "metric_name": METRIC, "message": "gap widened"},
            OutputFormat.MARKDOWN,
        )
    )
    detail = rep.sections[0]["content"]

    assert "Affected Groups: []" not in detail, detail
    assert "Affected Groups: not recorded" in detail, detail


def test_control_an_alert_that_names_groups_prints_exactly_those_groups():
    """OVER-CORRECTION CONTROL. No group is invented and none is dropped."""
    rep = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_alert_report(
            {
                "severity": "HIGH",
                "metric_name": METRIC,
                "affected_groups": ["gender_female", "race_black"],
                "priority_score": 7.25,
                "message": "m",
            },
            OutputFormat.MARKDOWN,
        )
    )

    assert "Affected Groups: gender_female, race_black" in rep.sections[0]["content"]


def test_an_alert_with_no_severity_is_not_reported_as_a_blank_verdict():
    """`d.get('severity', '?')` does not fire for a key present holding None, so
    the page printed "Severity: None"."""
    rep = _quiet(
        lambda: ReportGenerator(MetricsStore()).generate_alert_report(
            {"severity": None, "metric_name": METRIC, "message": "m"},
            OutputFormat.MARKDOWN,
        )
    )

    assert "Severity: UNSCORED (not determined)" in rep.sections[0]["content"]


# ===== the embedded charts


class _RaisingDashboard:
    """A dashboard whose figures fail, as create_operational_view genuinely did
    (PlotlyKeyError on any grid holding an Indicator, before the
    exclude_empty_subplots fix)."""

    def _boom(self):
        raise RuntimeError("PlotlyKeyError: Invalid property 'xaxis' for Indicator")

    create_executive_view = _boom
    create_operational_view = _boom
    create_technical_view = _boom


class _WorkingDashboard:
    def __init__(self, store):
        self._inner = FairnessDashboard(store, CFG)

    def create_executive_view(self):
        return self._inner.create_executive_view()


def test_a_chart_that_crashed_is_not_rendered_as_a_report_without_charts():
    """BEFORE: `except Exception: charts_html = ""`, so the 3637-character HTML
    carried no figure and no mention of a failure: identical to a report
    configured with no dashboard at all."""
    html = _quiet(
        lambda: ReportGenerator(
            MetricsStore(), dashboard=_RaisingDashboard()
        ).generate_executive_report(OutputFormat.HTML)
    ).content

    assert "COULD NOT RENDER" in html
    assert "PlotlyKeyError" in html
    assert "not because there was nothing to plot" in html


def test_control_a_working_dashboard_still_embeds_its_figure_and_says_nothing():
    """OVER-CORRECTION CONTROL."""
    store = _clean_store()
    html = _quiet(
        lambda: ReportGenerator(
            store, dashboard=_WorkingDashboard(store)
        ).generate_executive_report(OutputFormat.HTML)
    ).content

    assert "COULD NOT RENDER" not in html
    assert "plotly" in html.lower(), "the figure did not reach the page at all"
