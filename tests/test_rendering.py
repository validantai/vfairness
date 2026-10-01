"""Render-smoke tests for every registered SVG template.

VI_A_GR_029: the manifest test only checks that template files EXIST. These
tests actually drive ``render_svg`` for each registered template and assert
the output is well-formed XML, is an ``<svg>`` document, and contains no
unrendered Jinja tokens. That catches template syntax errors, malformed
markup, and stray ``{{ }}`` / ``{% %}`` that file-existence checks miss.

Sample data strategy (matches the issue's "shared minimal dict plus
per-template overrides"):
    - Jinja's default Undefined prints as empty and iterates as empty, so a
      missing scalar renders fine and a loop over a missing collection renders
      nothing. Only arithmetic / range() on an undefined scalar raises.
    - `_render_template` therefore renders with a small base dict and, on each
      UndefinedError, injects a numeric sentinel for the named scalar and
      retries. This auto-fills exactly the scalars a template does math on.
    - `_STRUCTURAL` supplies the few templates that need a real structure
      (a non-empty list used as a divisor, an object accessed by attribute,
      an int used by range()).
The assertions still run against the REAL rendered output, so a genuinely
broken template fails loudly.
"""

from __future__ import annotations

import re
import xml.dom.minidom as minidom

import pytest

from vfairness._registry import CAPABILITY_REGISTRY

# Skip the whole module if the rendering engine (jinja2) is unavailable.
engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)


REGISTERED_TEMPLATES = sorted(
    {t for e in CAPABILITY_REGISTRY.values() for t in e.get("svg_templates", [])}
)


class _Num(float):
    """A float that can also be used where an int is required (range())."""

    def __index__(self) -> int:  # noqa: D401 - dunder
        return int(self)


# Base data shared by every template. Strings are supplied so numeric
# auto-fill never lands on a display field that a string filter touches.
_BASE = {
    "title": "Sample Report",
    "subtitle": "Sample subtitle",
    "timestamp": "2026-01-01 00:00",
    "explanation": "",  # falsy -> the optional explanation block is skipped
    "recommendation": "Review flagged groups and re-evaluate.",
    "model_name": "Sample Model",
    "model_version": "v1",
    "attribute_name": "gender",
    "feature_name": "feature",
    "metric_name": "demographic_parity_difference",
    "task_type": "classification",
    "protected_attribute": "gender",
}

# Per-template structural data: non-empty divisors, attribute-accessed objects,
# range() ints, and lists sliced/measured at the top level.
_STRUCTURAL = {
    "bias_audit": {
        # `608 // modules|length` divides by len(modules): must be non-empty.
        "modules": [
            {
                "bg": "#eef2ff",
                "color": "#4f46e5",
                "name": "Classification",
                "score": 0.42,
                "detail": "2 findings",
            },
        ],
    },
    "cicd_pipeline": {
        "validation": {"issues": []},
        "gate": {"reasons": [], "warnings": []},
    },
    "confidence_intervals": {"n_ticks": 5},
    "threshold_optimization_report": {
        "fairness_improvement": {"original": 0.30, "optimized": 0.12, "reduction_pct": 60.0},
        "accuracy_change": {"original": 0.91, "optimized": 0.88, "change": -0.03},
    },
    "proxy_risk": {"features": []},
    "transformation_comparison": {"features": []},
    # The next two are not registry-referenced yet (2026-08-02) but the
    # parametrization below is dynamic over the registry, so the structural
    # data is staged here: the moment a registry entry references them, the
    # smoke test must render them instead of failing on attribute access
    # against an auto-filled numeric sentinel. Direct render tests live in
    # tests/test_audit_wave5_hardening.py.
    "intersectional_disparity": {
        "groups": [
            {
                "name": "A x young",
                "rate": 0.6,
                "n": 40,
                "rank": 1,
                "ground_truth": 0.55,
                "delta_pp": 0.0,
                "bar_color": "#334155",
                "dot_color": "#334155",
            },
            {
                "name": "B x old",
                "rate": 0.3,
                "n": 35,
                "rank": 2,
                "ground_truth": 0.5,
                "delta_pp": -30.0,
                "bar_color": "#7f1d1d",
                "dot_color": "#7f1d1d",
            },
        ],
        "insights": [{"bg": "#fef2f2", "color": "#7f1d1d", "text": "Gap of 30pp"}],
        "disadvantaged": {"group": "B x old", "rate": 0.3, "n": 35, "actual": 11, "predicted": 21},
        "privileged": {"group": "A x young", "rate": 0.6, "n": 40, "actual": 22, "predicted": 24},
    },
    "workflow_overview": {
        # `stages` and `lines` are {% set %} inside the template itself;
        # only these three lists come from context.
        "integrations": [
            {"name": "CI", "description": "gate", "color": "#334155", "status": "active"}
        ],
        "vcs_tools": [{"name": "git", "description": "hooks"}],
        "test_types": [{"name": "unit", "description": "metrics"}],
    },
}

_UNDEF_RE = re.compile(r"'([\w]+)' is undefined")


def _render_template(name: str) -> str:
    """Render a template, auto-filling undefined numeric scalars."""
    from jinja2.exceptions import UndefinedError

    data = dict(_BASE)
    data.update(_STRUCTURAL.get(name, {}))
    last = None
    for _ in range(400):
        try:
            return engine.render_svg(name, dict(data))
        except UndefinedError as exc:  # missing scalar used in arithmetic
            m = _UNDEF_RE.search(str(exc))
            if not m or m.group(1) in data:
                raise
            data[m.group(1)] = _Num(1.0)
            last = exc
    raise AssertionError(f"{name}: did not converge, last error: {last}")


def test_registered_templates_present():
    """Guard against an empty parametrization silently passing."""
    assert len(REGISTERED_TEMPLATES) >= 40


@pytest.mark.parametrize("name", REGISTERED_TEMPLATES)
def test_template_renders_wellformed_svg(name):
    out = _render_template(name)

    # 1. Non-empty and is an SVG document.
    assert out and out.strip(), f"{name}: rendered empty output"
    stripped = out.lstrip()
    if stripped.startswith("<?xml"):
        stripped = stripped[stripped.index("?>") + 2 :].lstrip()
    assert stripped.startswith("<svg"), (
        f"{name}: output does not start with <svg (got: {stripped[:40]!r})"
    )

    # 2. Parses as well-formed XML.
    try:
        minidom.parseString(out)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{name}: rendered output is not well-formed XML: {exc}")

    # 3. No unrendered Jinja tokens leaked through.
    assert "{{" not in out, f"{name}: leftover '{{{{' expression token"
    assert "{%" not in out, f"{name}: leftover '{{%' statement token"


# ── Radar fairness-axis semantics (fair = outward) ──────────────────────────


def test_radar_fairness_radius_direction():
    """The radar fairness score runs 1.0 at the rim (fair) to 0.0 at the centre
    (unfair), and every metric's threshold lands on the same ring, honouring the
    metric's direction."""
    adapters = pytest.importorskip("vfairness.rendering.adapters_fairness")
    frac = adapters._fairness_radius_fraction
    ring = adapters._THRESHOLD_RING_FRACTION

    # Difference metric (lower-is-better): parity -> rim, threshold -> ring,
    # twice the threshold -> centre; a fair value sits outside the ring, a
    # failing value inside it.
    assert frac("demographic_parity_difference", 0.0, 0.10) == 1.0
    assert frac("demographic_parity_difference", 0.10, 0.10) == pytest.approx(ring)
    assert frac("demographic_parity_difference", 0.20, 0.10) == 0.0
    assert frac("demographic_parity_difference", 0.02, 0.10) > ring
    assert frac("demographic_parity_difference", 0.22, 0.15) < ring

    # Ratio metric (higher-is-better, four-fifths 0.80): parity -> rim, the 0.80
    # threshold -> ring, a shortfall caves inward.
    assert frac("demographic_parity_ratio", 1.0, 0.80) == 1.0
    assert frac("demographic_parity_ratio", 0.80, 0.80) == pytest.approx(ring)
    assert frac("demographic_parity_ratio", 0.72, 0.80) < ring


def test_radar_svg_plots_fair_metrics_outward():
    """End-to-end: in the rendered SVG a fair metric's vertex is farther from the
    centre than the pass ring, and a failing metric's vertex is inside it."""
    import math

    adapters = pytest.importorskip("vfairness.rendering.adapters_fairness")
    report = {
        "metrics": {
            "demographic_parity_difference": 0.02,  # fair -> outward
            "predictive_parity_difference": 0.05,  # fair -> outward
            "equalized_odds_difference": 0.22,  # fail -> inward
        },
        "thresholds_used": {
            "demographic_parity_difference": 0.10,
            "predictive_parity_difference": 0.10,
            "equalized_odds_difference": 0.15,
        },
        "assessment": {"fairness_score": 0.5},
    }
    svg = adapters.radar_chart_to_svg(report)

    ring_px = adapters._THRESHOLD_RING_FRACTION * 140
    cx, cy = 340, 362
    poly = re.search(r'<polygon points="([0-9.,\s]+)" fill="[^"]+" fill-opacity="0.12"', svg)
    assert poly, "data polygon not found in radar SVG"
    coords = [tuple(map(float, p.split(","))) for p in poly.group(1).split()]
    dists = [math.hypot(x - cx, y - cy) for x, y in coords]

    assert dists[0] > ring_px, (
        f"fair metric should sit outside ring: {dists[0]:.0f} vs {ring_px:.0f}"
    )
    assert dists[1] > ring_px, (
        f"fair metric should sit outside ring: {dists[1]:.0f} vs {ring_px:.0f}"
    )
    assert dists[2] < ring_px, (
        f"failing metric should sit inside ring: {dists[2]:.0f} vs {ring_px:.0f}"
    )
