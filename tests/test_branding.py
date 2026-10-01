"""The validant.ai mark is one switch, and it covers every surface (VB-COM-2).

The offer page promises that removing the branding from generated SVGs and
reports is possible. These tests are what makes that promise checkable:

* the resolution order (explicit call > environment > branded default),
* a STRUCTURAL proof that every chart template draws its mark from the two
  guarded symbols, so no template can quietly keep branding the output, and
* a BEHAVIOURAL proof that the rendered bytes carry no "validant" anywhere
  once the switch is off, for charts, HTML reports, Markdown reports and
  compliance model cards.

The structural test matters because only 13 of the 44 templates render from an
empty data dict, so a purely behavioural sweep would silently cover a third of
them and report success.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest

import vfairness
from vfairness.branding import ENV_VAR, branding_enabled, set_branding
from vfairness.rendering import engine

BRAND_TOKEN = "validant"


@pytest.fixture(autouse=True)
def _clean_branding_state(monkeypatch):
    """No test may leak a branding override or env var into another."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    set_branding(None)
    yield
    set_branding(None)


# Resolution order


def test_default_is_branded():
    assert branding_enabled() is True


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "No", " none "])
def test_env_var_removes_the_mark(monkeypatch, value):
    monkeypatch.setenv(ENV_VAR, value)
    assert branding_enabled() is False


@pytest.mark.parametrize("value", ["on", "1", "true", "YES"])
def test_env_var_can_state_branded_explicitly(monkeypatch, value):
    monkeypatch.setenv(ENV_VAR, value)
    assert branding_enabled() is True


def test_unrecognised_env_value_warns_and_stays_branded(monkeypatch):
    """A typo must never silently strip the mark and leave you believing it worked."""
    monkeypatch.setenv(ENV_VAR, "disabled")
    with pytest.warns(UserWarning, match="not a recognised value"):
        assert branding_enabled() is True


def test_explicit_call_beats_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "off")
    set_branding(True)
    assert branding_enabled() is True
    set_branding(False)
    assert branding_enabled() is False


def test_none_clears_the_override_back_to_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "off")
    set_branding(True)
    assert branding_enabled() is True
    set_branding(None)
    assert branding_enabled() is False


def test_switch_is_exported_from_the_package():
    assert vfairness.branding_enabled is branding_enabled
    assert vfairness.set_branding is set_branding


# Structural: the guard covers every template


def _chart_templates():
    return [n for n in engine.list_templates() if not n.startswith("_")]


def test_every_chart_template_draws_its_mark_from_the_guarded_symbols():
    """No template may hardcode the mark, or the switch would not reach it."""
    missing = []
    for name in _chart_templates():
        source = engine.get_template_path(name).read_text(encoding="utf-8")
        if "validant-logo" not in source and "validant-icon" not in source:
            missing.append(name)
    assert not missing, f"templates carry no <use> of the guarded symbols: {missing}"


def test_no_template_defines_the_mark_itself():
    """The symbols live in _shared_defs.svg only, which is what the guard wraps."""
    offenders = [
        name
        for name in _chart_templates()
        if re.search(
            r'<symbol\s+id="validant-', engine.get_template_path(name).read_text(encoding="utf-8")
        )
    ]
    assert not offenders, f"templates define the mark outside the guard: {offenders}"


# Behavioural: rendered output


def _renderable_templates():
    out = []
    for name in _chart_templates():
        try:
            engine.render_svg(name, {})
        except Exception:
            continue
        out.append(name)
    return out


def test_some_templates_render_from_empty_data():
    """Guards the two sweeps below against silently asserting over nothing."""
    assert len(_renderable_templates()) >= 10


def test_branded_render_carries_the_mark():
    for name in _renderable_templates():
        assert 'id="validant-logo"' in engine.render_svg(name, {}), name


def test_unbranded_render_carries_no_trace_of_the_mark():
    set_branding(False)
    for name in _renderable_templates():
        svg = engine.render_svg(name, {})
        assert BRAND_TOKEN not in svg.lower(), f"{name} still carries the mark"


def test_unbranded_render_leaves_no_dangling_reference():
    """Guarding the defs alone is not enough: the <use> pointing at the absent
    id would still sit in the markup of a file handed to a client."""
    set_branding(False)
    for name in _renderable_templates():
        assert "<use" not in engine.render_svg(name, {}), f"{name} kept a dangling <use>"


def test_a_single_render_can_override_the_switch():
    """Per-call data wins, so one chart can stay branded while the rest are not."""
    set_branding(False)
    name = _renderable_templates()[0]
    assert 'id="validant-logo"' in engine.render_svg(name, {"branding": True})


def test_unbranded_svg_is_still_well_formed_xml():
    """Dropping the symbols must leave a valid document, not a broken one."""
    from xml.etree import ElementTree

    set_branding(False)
    for name in _renderable_templates():
        ElementTree.fromstring(engine.render_svg(name, {}))


# Behavioural: reports


def _health_score():
    from vfairness.operations.reporting.store import HealthScore

    return HealthScore(
        score=72.0,
        status="yellow",
        trend="flat",
        trend_slope=0.0,
        components={"metric_compliance": 70.0, "alert_frequency": 80.0, "drift_stability": 66.0},
        timestamp=datetime(2026, 1, 1, 12, 0),
        explanation="fixture",
        n_metrics=3,
        n_alerts=0,
    )


def _generator():
    from vfairness.operations.reporting.reports import ReportGenerator

    # The render methods read only self.config, never the store.
    return ReportGenerator(store=None)


def _render_report(fmt: str) -> str:
    from vfairness.operations.reporting.reports import ReportTier

    gen = _generator()
    method = gen._render_html if fmt == "html" else gen._render_markdown
    return method(_health_score(), [], [], ReportTier.EXECUTIVE, timedelta(days=7))


@pytest.mark.parametrize("fmt", ["html", "markdown"])
def test_report_is_branded_by_default(fmt):
    assert BRAND_TOKEN in _render_report(fmt).lower()


@pytest.mark.parametrize("fmt", ["html", "markdown"])
def test_report_drops_the_mark_when_switched_off(fmt):
    set_branding(False)
    assert BRAND_TOKEN not in _render_report(fmt).lower()


def test_html_report_keeps_its_timestamp_when_unbranded():
    """The timestamp is report metadata, not branding, so it must survive."""
    set_branding(False)
    html = _render_report("html")
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", html)


def _model_card() -> str:
    from vfairness.operations.reporting.compliance import generate_model_card

    return generate_model_card(
        system_profile={"name": "fixture", "purpose": "test", "version": "1.0"},
        metric_results={},
    )


def test_model_card_is_branded_by_default():
    assert BRAND_TOKEN in _model_card().lower()


def test_model_card_drops_the_mark_when_switched_off():
    set_branding(False)
    assert BRAND_TOKEN not in _model_card().lower()
