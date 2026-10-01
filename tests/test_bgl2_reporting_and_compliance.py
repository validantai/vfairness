"""The reporting and compliance surfaces that nothing had ever executed.

Nine public units in ``operations.reporting`` had no assertion behind them: three
dashboard views, two interactive-dashboard entry points, three report serialisers
and the ISO 42001 evidence map. They are the surfaces a reader of a compliance
report actually sees, which makes an unmeasured value published there worse than
the same value in a library internal.

WHAT WAS FOUND, and it is why this file leads with the compliance map.
``generate_iso42001_evidence_map({})`` reported **10.0% coverage of ISO 42001** for
a completely empty wizard. Three controls were paying out: A.2.4 and A.6.4 were
hardcoded ``"partial"`` with no evidence test of any kind, and A.6.3 returned
``"partial"`` in precisely the branch where no intervention evidence existed.
``coverage_percent`` credited every partial at half weight, so zero evidence bought
(0 + 3 x 0.5) / 15 = 10%. On a compliance artefact that number is the whole point of
the document.

The two controls that no wizard data can settle are now ``"manual"``, a fourth
state meaning "outside what this map determines", and they are excluded from the
percentage on both sides of the fraction rather than scored as half a success.
A.6.3's else-branch is a gap, because absence of evidence is a gap.

The rest of the stack was examined the same way and found HONEST, which is
recorded here so it stays that way: on an empty store the health score is
``None`` with status ``not_assessed`` and the explanation "This is not a score of
100 and certifies nothing", and the gauge carries ``value=None`` rather than a
zero that reads as a measurement.
"""

from __future__ import annotations

import pathlib
import warnings
from datetime import datetime, timedelta

import pandas as pd
import pytest

from vfairness import MetricsStore
from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map
from vfairness.operations.reporting.dashboard import FairnessDashboard
from vfairness.operations.reporting.interactive import InteractiveDashboard
from vfairness.operations.reporting.reports import ReportGenerator

FULL_WIZARD = {
    "system_profile": {"purpose": "credit scoring"},
    "bias_results": [{"finding": "disparity"}],
    "metric_results": {"demographic_parity": 0.12},
    "risk_register": [{"risk": "proxy discrimination"}],
    "monitoring_config": {"enabled": True},
    "intervention_results": [{"intervention": "threshold shift"}],
    "dpia": {"completed": True},
    "annex_iv": {"completed": True},
    "model_card": {"completed": True},
}


def _populated_store(n: int = 48) -> MetricsStore:
    now = datetime.now()
    rows = [
        {
            "timestamp": now - timedelta(hours=i * 0.5),
            "metric": "demographic_parity" if i % 2 == 0 else "equal_opportunity",
            "value": 0.30 if i % 3 == 0 else 0.03,
            "group": "female" if i % 2 == 0 else "male",
            "alert": i % 3 == 0,
        }
        for i in range(n)
    ]
    store = MetricsStore()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(pd.DataFrame(rows), group_col="group", alert_col="alert")
    return store


# ---------------------------------------------------------------------------
# ISO 42001 evidence map: no evidence must buy no credit
# ---------------------------------------------------------------------------


def test_an_empty_wizard_earns_zero_iso42001_coverage():
    """The defect: an empty wizard reported 10.0% coverage of the standard."""
    evidence = generate_iso42001_evidence_map({})

    assert evidence["coverage_percent"] == 0.0, (
        "An empty wizard bought ISO 42001 coverage. Controls that no wizard data "
        "can settle must not be credited at half weight."
    )
    assert evidence["covered"] == 0
    assert evidence["partial"] == 0


def test_controls_this_map_cannot_assess_are_named_and_excluded():
    """Three states on a compliance surface: covered, gap, and cannot-determine."""
    evidence = generate_iso42001_evidence_map({})

    manual = [c for c in evidence["controls"] if c["coverage_status"] == "manual"]
    assert manual, "the cannot-determine state disappeared from the evidence map"
    assert evidence["manual"] == len(manual)

    # Excluded from the denominator too, so the percentage is of what was assessable.
    assert evidence["assessable_controls"] == evidence["total_controls"] - evidence["manual"]
    assert str(evidence["assessable_controls"]) in evidence["coverage_basis"]

    for control in manual:
        assert "cannot determine" in control["evidence_description"].lower(), (
            f"{control['control_id']} is excluded from scoring but does not say so "
            f"where a reader of the control would see it"
        )

    # A.2.4 and A.6.4 are the two that had no evidence test at all.
    assert {c["control_id"] for c in manual} >= {"A.2.4", "A.6.4"}


def test_absent_intervention_evidence_is_a_gap_not_a_partial():
    """A.6.3's else-branch credited the absence of evidence."""
    control = next(
        c for c in generate_iso42001_evidence_map({})["controls"] if c["control_id"] == "A.6.3"
    )
    assert control["coverage_status"] == "gap"


def test_control_a_full_wizard_still_earns_coverage():
    """The over-correction control. A map that credits nothing would pass the tests
    above, so this asserts the score RESPONDS to real evidence."""
    empty = generate_iso42001_evidence_map({})
    full = generate_iso42001_evidence_map(FULL_WIZARD)

    assert full["coverage_percent"] > empty["coverage_percent"]
    assert full["coverage_percent"] > 50.0, (
        "a fully populated wizard should clear half the assessable controls; "
        f"got {full['coverage_percent']}"
    )
    assert full["covered"] > 0
    assert full["gaps"] == 0


def test_the_manual_controls_are_the_same_under_any_input():
    """A control excluded because the map cannot determine it is excluded whatever
    the wizard says. If evidence could move it, it was never a manual control."""
    empty = generate_iso42001_evidence_map({})
    full = generate_iso42001_evidence_map(FULL_WIZARD)
    assert empty["manual"] == full["manual"]


# ---------------------------------------------------------------------------
# Dashboard views
# ---------------------------------------------------------------------------


def test_dashboard_explanation_refuses_a_score_on_an_empty_store():
    explanation = FairnessDashboard(MetricsStore()).get_explanation()
    assert "could not be computed" in explanation
    assert "/100" not in explanation


def test_dashboard_explanation_reports_a_score_when_there_are_records():
    """Control: the refusal above is not a refusal to ever answer."""
    explanation = FairnessDashboard(_populated_store()).get_explanation()
    assert "/100" in explanation


def test_technical_view_leaves_the_gauge_empty_rather_than_showing_zero():
    """An indicator reading 0 is a measurement. None is the absence of one."""
    figure = FairnessDashboard(MetricsStore()).create_technical_view()
    indicators = [t for t in figure.data if t.type == "indicator"]
    assert indicators, "the technical view lost its health gauge"
    for trace in indicators:
        assert trace.value is None, (
            "an empty store produced a health gauge with a numeric value, which a "
            "reader cannot distinguish from a measured score"
        )


def test_technical_view_shows_a_value_when_there_are_records():
    figure = FairnessDashboard(_populated_store()).create_technical_view()
    indicators = [t for t in figure.data if t.type == "indicator"]
    assert any(t.value is not None for t in indicators)


def test_technical_view_honours_a_metric_subset():
    dashboard = FairnessDashboard(_populated_store())
    assert dashboard.create_technical_view(metrics=["demographic_parity"]).data


def test_subgroup_sunburst_renders_nothing_for_a_metric_it_does_not_hold():
    figure = FairnessDashboard(_populated_store()).create_subgroup_sunburst("no_such_metric")
    assert len(figure.data) == 0, (
        "an unknown metric produced sunburst wedges, which would show a reader a "
        "breakdown of data the store does not have"
    )
    assert figure.layout.annotations, "nothing told the reader why the chart is empty"


def test_subgroup_sunburst_renders_for_a_metric_it_does_hold():
    figure = FairnessDashboard(_populated_store()).create_subgroup_sunburst("demographic_parity")
    assert len(figure.data) > 0


# ---------------------------------------------------------------------------
# Report generation and serialisation
# ---------------------------------------------------------------------------


def test_a_report_over_no_records_carries_the_refusal_into_the_document():
    """The score, the dict and the rendered HTML must all say the same thing."""
    report = ReportGenerator(MetricsStore(), dashboard=FairnessDashboard(MetricsStore())).generate()

    assert report.health_score.score is None
    assert report.health_score.status == "not_assessed"

    payload = report.to_dict()
    assert payload["health_score"]["score"] is None
    assert payload["health_score"]["status"] == "not_assessed"
    assert "certifies nothing" in payload["health_score"]["explanation"]

    # The serialiser must not drop the disclosure on the way to the reader.
    assert "could not be computed" in report.content


def test_a_report_over_real_records_carries_a_score(tmp_path: pathlib.Path):
    store = _populated_store()
    report = ReportGenerator(store, dashboard=FairnessDashboard(store)).generate()

    assert report.health_score.score is not None
    payload = report.to_dict()
    assert payload["health_score"]["score"] == report.health_score.score
    assert payload["title"] == report.title
    assert payload["tier"] and payload["format"]

    destination = tmp_path / "report.html"
    report.save(str(destination))
    assert destination.exists()
    written = destination.read_text(encoding="utf-8")
    assert written == report.content, "save() wrote something other than the report"


# ---------------------------------------------------------------------------
# Interactive dashboard
# ---------------------------------------------------------------------------


def test_save_html_writes_a_self_contained_file(tmp_path: pathlib.Path):
    store = _populated_store()
    destination = tmp_path / "interactive.html"
    returned = InteractiveDashboard(store, dashboard=FairnessDashboard(store)).save_html(
        str(destination)
    )

    assert returned == str(destination)
    assert destination.exists()
    body = destination.read_text(encoding="utf-8")
    assert body.lstrip().lower().startswith("<!doctype html")
    assert "plotly" in body.lower(), "the standalone file depends on a served asset"


def test_run_passes_the_host_and_port_it_was_given(monkeypatch: pytest.MonkeyPatch):
    """``run`` is executed here, not started: the Dash server is replaced by a
    recorder. Without this the method had never run at all, and a typo in the
    argument names would have surfaced only on somebody's first launch."""
    store = _populated_store()
    interactive = InteractiveDashboard(store, dashboard=FairnessDashboard(store))
    calls: list[dict] = []

    class _Recorder:
        def run(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(interactive, "_app", _Recorder(), raising=False)
    interactive.run(port=9123, host="0.0.0.0")

    assert calls == [{"host": "0.0.0.0", "port": 9123, "debug": interactive.config.dash_debug}]
