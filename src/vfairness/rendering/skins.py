"""
SVG skin: the Blanco design language applied to every vfairness visualisation.

Blanco is the platform's one design language. It is applied to every SVG that
vfairness renders from computation results (reports, charts, gallery), there is
no other scheme. Blanco is "The Silent Gallery" as the platform's own theme
specification names it: sharp zero-radius corners, editorial type (Plus Jakarta Sans
body, JetBrains Mono for data), desaturated graphite/slate chrome, and colour
reserved for genuine data signal, which collapses to four semantic tones, pass
(green), warn/error (one terracotta hue), info (ocean blue) and neutral (slate).

Design decision: why a post-render transform, not re-templating
----------------------------------------------------------------
Every colour that ends up in a rendered SVG arrives as a hex literal in the
final markup, whether it originated in a Jinja template or was computed by an
adapter. Blanco therefore operates as a single post-render transform over the
output string (recolour + zero-radius + editorial type): it never has to touch
the ~45 templates or ~15 adapter modules.

Public API
----------
    >>> from vfairness.rendering import render_svg, apply_skin
    >>> svg = report.to_svg()             # already Blanco (the default)
    >>> svg = render_svg("bias_audit", data)   # Blanco
    >>> svg = apply_skin(raw_svg)         # apply Blanco to an existing SVG

``render_svg`` applies Blanco by default, so reports and adapters produce Blanco
with no extra step. ``apply_skin`` is idempotent on already-Blanco output.

The Blanco palette is the authoritative platform Blanco theme (its ``--bl-*``
design tokens and its four-tone semantic colour scale):
    graphite fg  #2e4057   slate primary #4e6078   muted #5a6a78
    border #cdd4da         warm page bg  #fcfbfa    card  #ffffff
    pass  #41ba1b / text #1c6d00
    warn  terracotta hsl(14 52% 47%) ≈ #b6573a  (all reds/oranges/ambers)
    info  ocean #006686
    brand gradient  cyan #43c9fe → green #41ba1b
"""

from __future__ import annotations

import re
from typing import Dict, List

# ── Blanco recolour map: original palette hex → Blanco palette hex ──────────
#
# Keys are lowercase, fully-expanded 6-digit hex. Every chromatic colour that
# can appear in a rendered vfairness SVG (templates + adapters + gallery design
# proposals) is mapped, so no vivid original tone leaks through. Greyscale that
# already reads as Blanco is still remapped onto the exact Blanco slate ramp so
# the neutrals feel of-a-piece.
BLANCO_MAP: Dict[str, str] = {
    # ── Brand gradient: validant teal→green, remapped onto the Blanco
    #    cyan→green brand ramp (#43c9fe → #41ba1b), interpolated across stops ─
    "#0aafe3": "#43c9fe",
    "#16b1b4": "#43c1c9",
    "#2bb460": "#43ba7f",
    "#39b72c": "#42b93f",
    "#3fb818": "#41ba1b",
    "#444444": "#2e4057",  # wordmark "validant.ai" text fill (#444) → graphite
    # ── Neutral chrome → Blanco surface tiers + ink ramp ───────────────────
    #    Surface tiers are the spec's exact tokens (blanco-theme.md §2):
    #    surface #fcfbfa · container-low #f4f3f2 · container #eeeeed ·
    #    container-high #e9e8e7 · border hsl(210 10% 82%) ≈ #c9ced4.
    "#ffffff": "#ffffff",  # cards stay white (surface-lowest)
    "#fefefe": "#ffffff",
    "#fafbfc": "#fcfbfa",
    "#f8fafc": "#fcfbfa",
    "#f8f9fb": "#fcfbfa",
    "#f8f9fa": "#fcfbfa",
    "#f7f8fa": "#f4f3f2",
    "#f3f4f6": "#f4f3f2",
    "#f1f5f9": "#f4f3f2",
    "#f0f1f3": "#f4f3f2",
    "#e8ecf1": "#e9e8e7",
    "#e5e7eb": "#eeeeed",
    "#e2e8f0": "#eeeeed",
    "#d1d5db": "#c9ced4",
    "#cbd5e1": "#c9ced4",
    "#b0b8c4": "#aab2bb",
    "#9ca3af": "#93a0ab",
    "#94a3b8": "#93a0ab",
    "#6b7280": "#5a6a78",
    "#64748b": "#5a6a78",
    "#4b5563": "#5a6a78",
    "#475569": "#4e6078",
    "#374151": "#4e6078",
    "#334155": "#4e6078",
    "#2a3a50": "#2e4057",
    "#1e293b": "#2e4057",
    "#1a1a2e": "#2e4057",
    "#0f172a": "#2e4057",
    # ── Pass (green / fair / low-risk / on-target) → Blanco pass green ──────
    "#34d399": "#41ba1b",
    "#10b981": "#41ba1b",
    "#059669": "#41ba1b",
    "#16a34a": "#41ba1b",
    "#22c55e": "#41ba1b",
    "#047857": "#1c6d00",
    "#065f46": "#1c6d00",
    "#166534": "#1c6d00",
    "#a7f3d0": "#d6ecca",
    "#bbf7d0": "#d6ecca",
    "#86efac": "#c2e3b0",
    "#4ade80": "#8fce6f",
    "#d1fae5": "#e4f3da",
    "#dcfce7": "#e4f3da",
    "#ecfdf5": "#ecf8e8",
    "#f0fdf4": "#ecf8e8",
    # ── Warn / error: reds (high / fail / critical) → full terracotta ─────
    "#ef4444": "#b6573a",
    "#dc2626": "#b6573a",
    "#f87171": "#b6573a",
    "#b91c1c": "#8f4327",
    "#991b1b": "#8f4327",
    "#7f1d1d": "#7a3a24",
    "#fca5a5": "#e2bcb0",
    "#fecaca": "#efd9d2",
    "#fee2e2": "#f7ece8",
    "#fef2f2": "#faf1ee",
    "#fff8f8": "#faf1ee",
    # ── Warn: ambers / oranges → the SAME single terracotta hue as reds ───
    #    (spec §2: all reds, oranges and ambers collapse to one warn hue;
    #    intensity is carried by the tint background, not a second hue.)
    "#f97316": "#b6573a",
    "#ea580c": "#b6573a",
    "#f59e0b": "#b6573a",
    "#fbbf24": "#b6573a",
    "#d97706": "#8f4327",
    "#b45309": "#8f4327",
    "#92400e": "#8f4327",
    "#78350f": "#7a3a24",
    "#713f12": "#7a3a24",
    "#fed7aa": "#efd9d2",
    "#fde68a": "#efd9d2",
    "#fef3c7": "#f7ece8",
    "#fef9c3": "#f7ece8",
    "#fefce8": "#faf3ee",
    "#fffbeb": "#faf1ee",
    "#fff7ed": "#faf1ee",
    # ── Info: blues → Blanco ocean ────────────────────────────────────────
    "#3b82f6": "#006686",
    "#2563eb": "#006686",
    "#60a5fa": "#4f93a8",
    "#1d4ed8": "#004d63",
    "#1e40af": "#004d63",
    "#1e3a8a": "#004d63",
    "#93c5fd": "#a3c4cf",
    "#bfdbfe": "#cfe2e9",
    "#dbeafe": "#d4e6ec",
    "#eff6ff": "#ebf3f5",
    "#f0f9ff": "#ebf3f5",
    # ── Indigo / violet / purple → ocean (Blanco collapses these to info) ──
    "#8b5cf6": "#006686",
    "#7c3aed": "#004d63",
    "#6d28d9": "#004d63",
    "#4338ca": "#004d63",
    "#a855f7": "#006686",
    "#ede9fe": "#d4e6ec",
    "#e0e7ff": "#d4e6ec",
    # ── Pink / rose / fuchsia → terracotta (per the repo-wide Blanco sweep) ─
    "#ec4899": "#b6573a",
    "#db2777": "#8f4327",
    "#f472b6": "#b6573a",
    "#fce7f3": "#f7ece8",
}

# ── Skin registry ──────────────────────────────────────────────────────────
# Blanco is the platform's one design language; there is no other scheme.
SKINS: Dict[str, Dict[str, str]] = {
    "blanco": {
        "label": "Blanco",
        "description": (
            "The platform's Blanco design language ('The Silent Gallery'): "
            "sharp zero-radius corners, editorial type, desaturated "
            "graphite/slate chrome, and colour reserved for the four semantic "
            "tones (pass, warn, info, neutral)."
        ),
    },
}

DEFAULT_SKIN = "blanco"

# Matches a 3-, 4-, 6- or 8-digit hex colour token. Longest alternatives come
# first so ``#rrggbbaa`` is captured whole rather than as ``#rrggbb`` + ``aa``.
_HEX_RE = re.compile(
    r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})"
    r"(?![0-9a-fA-F])"
)

# Splits an SVG document into tag markup (``<...>``) and the text between
# tags, so recolouring can be limited to attribute values and <style> CSS and
# never rewrites hex codes that appear in visible text content.
_TAG_SPLIT_RE = re.compile(r"(<[^>]*>)")
_STYLE_OPEN_RE = re.compile(r"^<style(?![\w-])", re.IGNORECASE)
_STYLE_CLOSE_RE = re.compile(r"^</style", re.IGNORECASE)

# ── Structural Blanco transforms ("The Silent Gallery") ─────────────────────
# Blanco is not only a palette. Its two most load-bearing rules are structural:
#   1. Zero border-radius everywhere: sharp, architectural corners, no
#      exceptions (blanco-theme.md §1, §4). The templates round corners only on
#      <rect> (there are no <ellipse> elements), so dropping every rx/ry safely
#      squares every tile, badge, card, pill and bar.
#   2. Editorial type: body text renders in Plus Jakarta Sans; numeric and
#      technical values already render in JetBrains Mono (spec: mono for
#      anything countable), which is left untouched.
_RADIUS_RE = re.compile(r'\s+(?:rx|ry)="[^"]*"')
_INTER_FONT_RE = re.compile(r'''font-family="'Inter'[^"]*"''')
_BLANCO_SANS = (
    "font-family=\"'Plus Jakarta Sans','Inter','Segoe UI',system-ui,-apple-system,sans-serif\""
)


def _blanco_structure(svg: str) -> str:
    """Apply Blanco's non-colour rules: zero border-radius + editorial sans."""
    svg = _RADIUS_RE.sub("", svg)
    svg = _INTER_FONT_RE.sub(_BLANCO_SANS, svg)
    return svg


def list_skins() -> List[str]:
    """Return the available skin names.

    Blanco is the only one. The docstring promised "``original`` first" long
    after the ``original`` skin was removed, i.e. it named a retired surface and
    an ordering over a set of one (G13, 2026-09-30).

    A NEW list each call, deliberately: handing back ``SKINS.keys()`` would be a
    live view of the module's own registry, so a caller who mutated the result
    would be editing it.
    """
    return list(SKINS.keys())


def _split_hex(token: str) -> "tuple[str, str]":
    """
    Split a ``#hex`` token into ``(base6, alpha)``.

    ``base6`` is a lowercased ``#rrggbb`` used for palette lookup; ``alpha`` is
    the two-char opacity suffix (from ``#rrggbbaa`` / ``#rgba``) or ``''``.
    Translucent tints therefore recolour their RGB base while keeping opacity.
    """
    h = token[1:].lower()
    if len(h) == 3:  # #rgb
        return "#" + "".join(c * 2 for c in h), ""
    if len(h) == 4:  # #rgba
        return "#" + "".join(c * 2 for c in h[:3]), h[3] * 2
    if len(h) == 8:  # #rrggbbaa
        return "#" + h[:6], h[6:]
    return "#" + h, ""  # #rrggbb


def apply_skin(svg: str, skin: str = DEFAULT_SKIN) -> str:
    """
    Recolour a rendered SVG string into the requested *skin*.

    Parameters
    ----------
    svg : str
        A rendered SVG document (the output of :func:`render_svg`).
    skin : str
        Skin name. Only ``"blanco"`` (the default) is supported. Applying it to
        an already-Blanco SVG is idempotent.

    Returns
    -------
    str
        The Blanco SVG: recoloured, zero border-radius, editorial type. Colours
        not present in the palette map are left untouched, so structural markup
        and gradients survive intact.

    Raises
    ------
    ValueError
        If *skin* is not a registered skin name.
    """
    if skin not in SKINS:
        raise ValueError(f"Unknown skin '{skin}'. Available: {', '.join(list_skins())}")

    def _sub(match: "re.Match[str]") -> str:
        token = match.group(0)
        base, alpha = _split_hex(token)
        mapped = BLANCO_MAP.get(base)
        return token if mapped is None else mapped + alpha

    # Recolour only markup contexts: tag attributes and CSS inside <style>
    # blocks. Text content between tags (e.g. a chart caption quoting a hex
    # code) is user-visible data and must never be rewritten.
    parts = _TAG_SPLIT_RE.split(svg)
    in_style = False
    out = []
    for part in parts:
        if part.startswith("<"):
            out.append(_HEX_RE.sub(_sub, part))
            if _STYLE_OPEN_RE.match(part) and not part.endswith("/>"):
                in_style = True
            elif _STYLE_CLOSE_RE.match(part):
                in_style = False
        else:
            out.append(_HEX_RE.sub(_sub, part) if in_style else part)
    recoloured = "".join(out)
    return _blanco_structure(recoloured)
