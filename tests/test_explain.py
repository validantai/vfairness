"""Tests for the SVG explanation + accessibility layer (rendering.explain).

The core is pure stdlib, so most tests run without jinja2/numpy. The engine
integration test is guarded on jinja2.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPLAIN_PY = REPO_ROOT / "src" / "vfairness" / "rendering" / "explain.py"
TEMPLATE_DIR = REPO_ROOT / "src" / "vfairness" / "rendering" / "templates"


def _load_explain():
    spec = importlib.util.spec_from_file_location("vf_explain_under_test", EXPLAIN_PY)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the @dataclass under `from __future__ import
    # annotations` resolves its module via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


explain = _load_explain()

# Real template names (the strings render_svg receives), from the templates dir.
TEMPLATES = sorted(p.stem for p in TEMPLATE_DIR.glob("*.svg") if not p.stem.startswith("_"))


# ── coverage: every template is curated and has a finder ─────────────────────


def test_templates_discovered():
    assert len(TEMPLATES) >= 40, TEMPLATES


def test_every_template_has_chart_meta():
    missing = [t for t in TEMPLATES if t not in explain.CHART_META]
    assert missing == [], f"templates without CHART_META: {missing}"


def test_every_template_has_a_finder():
    missing = [t for t in TEMPLATES if t not in explain._FINDERS]
    assert missing == [], f"templates without a finding extractor: {missing}"


def test_chart_meta_entries_are_complete():
    for name, meta in explain.CHART_META.items():
        for key in ("title", "concept", "reading", "action"):
            assert meta.get(key), f"{name} missing {key}"


# ── build_explanation is bulletproof ─────────────────────────────────────────


@pytest.mark.parametrize("template", TEMPLATES)
def test_build_explanation_never_raises_on_empty_data(template):
    ce = explain.build_explanation(template, {})
    assert ce.title and ce.concept and ce.reading and ce.action
    assert ce.severity in explain._VALID_SEVERITY
    assert ce.paragraph().strip()
    assert ce.caption().strip()


@pytest.mark.parametrize("template", TEMPLATES)
def test_build_explanation_survives_hostile_data(template):
    for bad in (None, {}, {"anything": object()}, {"overall_score": "x"}, []):
        ce = explain.build_explanation(template, bad)
        assert ce.severity in explain._VALID_SEVERITY
        # metadata must be JSON-serialisable
        json.dumps(ce.metadata())


def test_unknown_template_falls_back_to_generic():
    ce = explain.build_explanation("no_such_template", {"subtitle": "hi"})
    assert ce.title == explain._GENERIC_META["title"]
    assert ce.severity in explain._VALID_SEVERITY


# ── severity agrees with the badge (the MEDIUM/MINIMAL bug) ───────────────────


def test_bias_audit_severity_matches_badge_label():
    minimal = explain.build_explanation(
        "bias_audit", {"overall_score": 0.12, "overall_label": "MINIMAL", "n_critical": 1}
    )
    assert "MINIMAL" in minimal.finding
    assert minimal.severity == "info"  # never "medium" next to a MINIMAL badge

    high = explain.build_explanation(
        "bias_audit", {"overall_score": 0.82, "overall_label": "HIGH", "n_critical": 3}
    )
    assert "HIGH" in high.finding and high.severity == "high"


def test_finding_severity_is_always_valid():
    for name in explain._FINDERS:
        _, sev = explain._finding_for(name, {})
        assert sev in explain._VALID_SEVERITY, (name, sev)


# ── accessibility injection ──────────────────────────────────────────────────

_SAMPLE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>'


def test_inject_accessibility_adds_role_title_desc_metadata():
    ce = explain.build_explanation("radar_chart", {"status_text": "Unfair"})
    out = explain.inject_accessibility(_SAMPLE_SVG, ce)
    assert 'role="img"' in out
    assert out.count("<title>") == 1 and out.count("<desc>") == 1
    assert 'id="vfairness-explanation"' in out
    # the embedded metadata JSON is valid and carries the finding + severity
    m = re.search(r"<!\[CDATA\[(.*?)\]\]>", out, re.S)
    payload = json.loads(m.group(1))
    assert payload["chart"] == "radar_chart"
    assert payload["severity"] in explain._VALID_SEVERITY
    assert payload["title"] and payload["howToRead"]


def test_inject_accessibility_noop_without_svg():
    ce = explain.build_explanation("radar_chart", {})
    assert explain.inject_accessibility("", ce) == ""
    assert explain.inject_accessibility("<div>not an svg</div>", ce) == "<div>not an svg</div>"


def test_inject_accessibility_preserves_existing_role():
    ce = explain.build_explanation("radar_chart", {})
    svg = '<svg role="presentation"><rect/></svg>'
    out = explain.inject_accessibility(svg, ce)
    assert 'role="presentation"' in out and 'role="img"' not in out


def test_desc_and_title_are_xml_escaped():
    # a finding with an ampersand/quote must not break the XML
    ce = explain.ChartExplanation(
        template="x",
        title="A & B",
        concept="c",
        reading="r",
        action="a",
        finding='has "quotes" & <angle>',
        severity="high",
    )
    out = explain.inject_accessibility(_SAMPLE_SVG, ce)
    assert "<A & B>" not in out and "&amp;" in out and "&lt;angle&gt;" in out


# ── engine integration (needs jinja2) ────────────────────────────────────────


def test_render_svg_auto_explains_and_describes():
    engine = pytest.importorskip("vfairness.rendering.engine", reason="engine needs jinja2")
    if not engine.JINJA2_AVAILABLE:  # pragma: no cover
        pytest.skip("jinja2 not installed")

    svg = engine.render_svg("radar_chart", {"status_text": "Unfair"})
    assert 'role="img"' in svg and "<desc>" in svg and "vfairness-explanation" in svg
    assert ">Explanation<" in svg  # auto-filled on-canvas panel

    suppressed = engine.render_svg("radar_chart", {"status_text": "Unfair", "explanation": ""})
    assert ">Explanation<" not in suppressed  # panel suppressed
    assert "<desc>" in suppressed  # accessibility still present
