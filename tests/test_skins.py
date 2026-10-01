"""Tests for the SVG skin system (``vfairness.rendering.skins``).

The core recolour logic is pure stdlib, so most tests run without jinja2/numpy.
The single engine-integration test is guarded on jinja2 availability.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKINS_PY = REPO_ROOT / "src" / "vfairness" / "rendering" / "skins.py"
GALLERY_DIR = REPO_ROOT / "docs" / "site" / "img" / "svg-gallery"

# Vivid original tones that must never survive a Blanco recolour.
VIVID_ORIGINALS = [
    "#dc2626",
    "#ef4444",
    "#b91c1c",
    "#f59e0b",
    "#f97316",
    "#059669",
    "#10b981",
    "#3b82f6",
    "#1d4ed8",
    "#8b5cf6",
    "#ec4899",
    "#0aafe3",
    "#3fb818",
    "#0f172a",
    "#94a3b8",
    "#64748b",
    "#f1f5f9",
]


def _load_skins():
    """Import skins.py directly (avoids the numpy-heavy top-level package)."""
    spec = importlib.util.spec_from_file_location("vf_skins_under_test", SKINS_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


skins = _load_skins()


# A compact SVG exercising every colour family, an alpha tint, a 3-digit hex,
# rounded corners (rx/ry), and both the Inter and JetBrains Mono font stacks.
SAMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<rect fill="#dc2626" rx="10" ry="8"/><rect fill="#f59e0b" rx="3"/>'
    '<rect fill="#059669"/><rect fill="#3b82f6"/><rect fill="#8b5cf6"/>'
    '<rect fill="#ec4899"/>'
    '<stop stop-color="#0aafe3"/><stop stop-color="#3fb818"/>'
    "<text font-family=\"'Inter','Segoe UI',system-ui,sans-serif\" fill=\"#0f172a\">a</text>"
    "<text font-family=\"'JetBrains Mono','SF Mono',monospace\" fill=\"#94a3b8\">0.84</text>"
    '<rect fill="#dc262620"/><path fill="#444"/>'
    "</svg>"
)


# ── registry ────────────────────────────────────────────────────────────────


def test_blanco_is_the_only_skin():
    # Blanco is the platform's one design language; there is no 'original'.
    assert skins.list_skins() == ["blanco"]
    assert skins.DEFAULT_SKIN == "blanco"


def test_skins_registry_has_metadata():
    for name in skins.list_skins():
        entry = skins.SKINS[name]
        assert entry["label"] and entry["description"]


# ── apply_skin: default + errors ────────────────────────────────────────────


def test_default_applies_blanco():
    # No argument => Blanco (not a no-op): the sample's rounded, vivid, Inter
    # markup comes back sharp, terracotta and Plus-Jakarta.
    out = skins.apply_skin(SAMPLE_SVG)
    assert "rx=" not in out
    assert "#dc2626" not in out.lower() and "#b6573a" in out.lower()
    assert "Plus Jakarta Sans" in out


def test_original_skin_is_removed():
    with pytest.raises(ValueError):
        skins.apply_skin(SAMPLE_SVG, "original")


def test_unknown_skin_raises():
    with pytest.raises(ValueError):
        skins.apply_skin(SAMPLE_SVG, "midnight")


# ── apply_skin: Blanco recolour ─────────────────────────────────────────────


def test_blanco_removes_all_vivid_originals():
    out = skins.apply_skin(SAMPLE_SVG, "blanco").lower()
    leaked = [h for h in VIVID_ORIGINALS if h in out]
    assert leaked == [], f"vivid originals leaked: {leaked}"


def test_blanco_maps_each_family_to_its_token():
    out = skins.apply_skin(SAMPLE_SVG, "blanco").lower()
    assert "#b6573a" in out  # red AND amber -> the one terracotta warn hue
    assert "#41ba1b" in out  # green -> pass
    assert "#006686" in out  # blue & purple -> ocean
    assert "#2e4057" in out  # slate-900 & #444 -> graphite
    assert "#43c9fe" in out  # brand gradient start -> Blanco cyan


def test_blanco_unifies_amber_and_red_to_one_hue():
    # spec §2: all reds, oranges and ambers collapse to a single terracotta hue
    red = skins.apply_skin('<rect fill="#dc2626"/>', "blanco")
    amber = skins.apply_skin('<rect fill="#f59e0b"/>', "blanco")
    assert red == '<rect fill="#b6573a"/>'
    assert amber == '<rect fill="#b6573a"/>'


def test_blanco_zeroes_border_radius():
    # spec §1/§4: zero border-radius everywhere, no exceptions
    out = skins.apply_skin('<rect fill="#0f172a" rx="10" ry="8"/>', "blanco")
    assert "rx=" not in out and "ry=" not in out
    assert '<rect fill="#2e4057"/>' == out


def test_blanco_swaps_inter_to_plus_jakarta_but_keeps_mono():
    out = skins.apply_skin(SAMPLE_SVG, "blanco")
    assert "Plus Jakarta Sans" in out
    assert "'Inter'" in out  # kept as a fallback
    assert "'JetBrains Mono'" in out  # data font untouched (spec: mono for values)


def test_blanco_is_idempotent():
    # Applying Blanco to already-Blanco output changes nothing.
    once = skins.apply_skin(SAMPLE_SVG, "blanco")
    twice = skins.apply_skin(once, "blanco")
    assert once == twice


def test_blanco_preserves_alpha_suffix():
    # #dc262620 (terracotta base at ~12% alpha) -> #b6573a20
    out = skins.apply_skin('<rect fill="#dc262620"/>', "blanco")
    assert out == '<rect fill="#b6573a20"/>'


def test_blanco_expands_and_maps_three_digit_hex():
    # #444 -> #444444 -> graphite #2e4057
    out = skins.apply_skin('<path fill="#444"/>', "blanco")
    assert out == '<path fill="#2e4057"/>'


def test_blanco_is_case_insensitive():
    out = skins.apply_skin('<rect fill="#DC2626"/>', "blanco")
    assert out == '<rect fill="#b6573a"/>'


def test_blanco_leaves_unmapped_hex_untouched():
    # A colour not in the palette map passes through unchanged.
    out = skins.apply_skin('<rect fill="#123456"/>', "blanco")
    assert out == '<rect fill="#123456"/>'


def test_blanco_output_uses_only_blanco_palette():
    """No chromatic tone outside the Blanco target set may appear."""
    import re

    out = skins.apply_skin(SAMPLE_SVG, "blanco")
    bases = {"#" + m.group(0)[1:7].lower() for m in re.finditer(r"#[0-9a-fA-F]{6,8}\b", out)}
    allowed = set(skins.BLANCO_MAP.values())
    stray = bases - allowed
    assert stray == set(), f"unexpected non-Blanco tones: {stray}"


# ── on-disk gallery: the shipped files are Blanco ───────────────────────────


@pytest.mark.skipif(not GALLERY_DIR.is_dir(), reason="gallery images not present")
def test_shipped_gallery_files_are_blanco():
    files = sorted(p for p in GALLERY_DIR.glob("*.svg") if p.parent == GALLERY_DIR)
    assert files, "no gallery SVGs found"
    for svg_path in files:
        svg = svg_path.read_text(encoding="utf-8")
        low = svg.lower()
        leaked = [h for h in VIVID_ORIGINALS if h in low]
        assert leaked == [], f"{svg_path.name}: vivid tones present: {leaked}"
        assert "rx=" not in svg, f"{svg_path.name}: rounded corners (rx) present"


# ── engine integration (needs jinja2) ───────────────────────────────────────


def test_render_svg_defaults_to_blanco():
    engine = pytest.importorskip("vfairness.rendering.engine", reason="engine needs jinja2")
    if not engine.JINJA2_AVAILABLE:  # pragma: no cover
        pytest.skip("jinja2 not installed")

    # Find any template that renders with a trivial data dict and assert the
    # default render is Blanco: sharp, no vivid tones, editorial sans.
    rendered = None
    for name in engine.list_templates():
        if name.startswith("_"):
            continue
        try:
            rendered = engine.render_svg(name, {})  # no skin arg => default
        except Exception:
            continue
        break

    if rendered is None:  # pragma: no cover
        pytest.skip("no template rendered with an empty data dict")

    low = rendered.lower()
    assert "rx=" not in rendered
    assert not [h for h in VIVID_ORIGINALS if h in low]
    # and it equals an explicit blanco render
    from vfairness.rendering.skins import apply_skin

    assert rendered == apply_skin(rendered, "blanco")
