"""
Template rendering engine for SVG report generation.

Jinja2 is REQUIRED, and it is the ``rendering`` extra
(``pip install 'vfairness[rendering]'``). There is no fallback renderer: this
docstring used to promise "a minimal built-in renderer", and no such path has
ever existed here (grep for string.Template: the only hit was that promise).
Without Jinja2 every entry point raises ``ImportError`` via
``raise_jinja2_missing`` below; none returns an empty string.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, NoReturn

from ..branding import branding_enabled

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Check Jinja2 availability at import time
try:
    import jinja2 as _jinja2  # noqa: F401

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


#: The one message every rendering entry point raises when Jinja2 is absent.
#: It names the EXTRA rather than the bare distribution, because that is how
#: this package declares the dependency (pyproject: rendering = ["jinja2>=3.0"])
#: and how the mcp surface already words its own hint.
JINJA2_MISSING_MESSAGE = (
    "Jinja2 is required for SVG report rendering. "
    "Install it with: pip install 'vfairness[rendering]'"
)


def raise_jinja2_missing() -> NoReturn:
    """Refuse to render without Jinja2, identically everywhere.

    Every ``*_to_svg`` adapter and ``render_svg`` funnels its missing-backend
    case through here. Until 2026-08-28, 16 of the 44 public adapters instead
    did ``warnings.warn(...); return ""``: their docstrings document the return
    as "SVG markup string", so a caller doing ``open(p, "w").write(svg)`` wrote a
    ZERO-BYTE file and read it as this run's report, while the other 28 raised.
    Python's default filter also shows a warning once per process, so a batch job
    warned once and then emitted nothing, silently, for every report after it.
    An empty string is not markup, and "could not render" must not be
    indistinguishable from "rendered nothing worth showing".

    Raises:
        ImportError: always. Callers guard the call with ``JINJA2_AVAILABLE``.
    """
    raise ImportError(JINJA2_MISSING_MESSAGE)


def get_template_path(name: str) -> Path:
    """Return the full path to a named template."""
    path = TEMPLATE_DIR / f"{name}.svg"
    if not path.exists():
        raise FileNotFoundError(
            f"Template '{name}' not found at {path}. Available: {', '.join(list_templates())}"
        )
    return path


def list_templates() -> List[str]:
    """List available template names (without extension)."""
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.svg"))


# Jinja2 environment (lazy init)
_jinja_env = None


def _get_jinja_env():
    """Create or return the cached Jinja2 environment."""
    global _jinja_env
    if _jinja_env is not None:
        return _jinja_env

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        return None

    _jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["svg"]),  # SVG is XML: escape &, <, >
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Custom filters
    def _to_float_local(v):
        """The number a numeric filter may print, or None for "N/A".

        CRITICAL, three states, never two, and this is the LAST door before the
        markup. Every numeric filter below prints ``_to_float(v)`` when it is not
        None, and until 2026-09-30 a NaN and an infinity both survived
        ``float(v)``: ``f"{float('nan'):.4f}"`` is the string ``"nan"``, so a
        quantity nobody could measure was drawn in the cell, in the font and in
        the colour of a measurement, on EVERY template at once. Executed on this
        tree, ``method_comparison_to_svg`` drew ``<text ...>nan</text>`` as a
        method's accuracy and ``training_analysis_report_to_svg`` drew "Base rate
        disparity: nan" in its subtitle. ``pct`` was worse again: it renders
        ``"nan%"``, which reads as a percentage.

        "N/A" is the string these filters already emit for None, so the third
        state has one spelling across the whole library rather than two. The
        adapters withhold such values at their own boundary as well; this is the
        net under them, for the hand-built and third-party dicts they also render.

        A bool is refused for the same reason ``adapters_training._coord``
        refuses one: ``True`` is an ``int`` in Python and would print as 1.0000,
        the best value on an accuracy axis.

        THE BODY LIVES AT MODULE LEVEL now (:func:`_to_float`), because two of
        the filters below were NOT going through this door and this docstring
        said every one of them was. See ``bar_width`` and ``truncate_text``.
        """
        return _to_float(v)

    _jinja_env.filters["pct"] = lambda v: (
        f"{_to_float(v):.0%}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["pct1"] = lambda v: (
        f"{_to_float(v):.1%}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["f1"] = lambda v: (
        f"{_to_float(v):.1f}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["f2"] = lambda v: (
        f"{_to_float(v):.2f}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["f3"] = lambda v: (
        f"{_to_float(v):.3f}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["f4"] = lambda v: (
        f"{_to_float(v):.4f}" if _to_float(v) is not None else "N/A"
    )
    _jinja_env.filters["risk_color"] = _risk_color
    _jinja_env.filters["risk_bg"] = _risk_bg
    _jinja_env.filters["risk_label"] = _risk_label
    _jinja_env.filters["bar_width"] = _bar_width
    _jinja_env.filters["severity_icon"] = _severity_icon
    _jinja_env.filters["truncate_text"] = _truncate_text
    _jinja_env.globals["max"] = max
    _jinja_env.globals["min"] = min
    _jinja_env.globals["abs"] = abs

    return _jinja_env


# Colour helpers (used as Jinja2 filters and by adapters)


def _to_float(v):
    """The number any filter here may print or measure with, or None.

    The body of ``_get_jinja_env._to_float``, lifted to module level on
    2026-09-30 so that it really is what its own docstring calls it: the LAST
    door before the markup, for EVERY filter. Two of them were not going
    through it.

    See the docstring inside :func:`_get_jinja_env` for the incident that
    created this rule: ``f"{float('nan'):.4f}"`` is the string ``"nan"`` and
    ``pct(nan)`` is ``"nan%"``, which reads as a percentage; and ``True`` is an
    ``int``, so ``f4(True)`` was ``"1.0000"``, the best value on an accuracy
    axis.
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        num = float(v)
    except (ValueError, TypeError):
        return None
    return num if math.isfinite(num) else None


def _bar_width(v, max_w=300):
    """Bar length in px for a 0-1 value, or 0 px when there is no value.

    A LENGTH IS A MEASUREMENT A READER COMPARES BY EYE, so it goes through the
    same door as the printed number beside it. Until 2026-09-30 this filter was
    ``min(max_w, max(4, v * max_w))`` on the RAW value, the one numeric filter
    in this module that skipped ``_to_float`` (``truncate_text`` was the other
    non-numeric one), and measured here:

    * ``bar_width(True)`` was ``300``, the FULL bar, i.e. the highest rate on
      the panel, while the label beside it printed "N/A" through ``pct1``.
      ``True`` is an ``int``; that is the whole reason ``_to_float`` refuses a
      bool.
    * ``bar_width(nan)`` was ``4``, indistinguishable from a measured rate of
      0.0, because ``nan > 4`` is False so ``max()`` returned the floor.
    * ``bar_width("0.5")`` raised ``TypeError``, so a serialised number took the
      whole chart down, and ``bar_width(None)`` likewise, although
      ``fairness_report.svg`` renders a None rate's LABEL as "N/A" perfectly
      well.

    ZERO, not the 4px floor, for an unmeasurable value: that gives three
    visually distinct states on one panel (no bar + "N/A", a 4px stub + "0.0%",
    a longer bar) instead of two. The 4px floor stays for a MEASURED zero,
    which is what it was for.
    """
    num = _to_float(v)
    if num is None:
        return 0
    return min(max_w, max(4, num * max_w))


def _truncate_text(s, n=50):
    """Trim a label to *n* characters, without MINTING one out of absence.

    ``str(x)`` on an absent value mints content, and this filter is applied to
    roughly ninety label, name and message slots across 32 templates. Measured
    2026-09-30: ``truncate_text(None)`` returned the string ``"None"`` and
    ``truncate_text(float('nan'))`` returned ``"nan"``, so a group, a feature or
    a finding whose name was absent got one, in the font and the position of a
    real label. ``pd.NA`` and ``pd.NaT`` print ``"<NA>"`` and ``"NaT"`` the same
    way; a group named ``'<NA>'`` carrying a maximal finding is already on
    record elsewhere in this library.

    "N/A" is the one spelling the numeric filters here already use for the third
    state, so a reader meets the same token everywhere.

    A STRING IS PASSED THROUGH UNCHANGED, including the literal "None". That is
    deliberate and it is the narrow half of the six-doors rule: this filter also
    renders free prose (issue messages, recommendations, insights), and a caller
    who supplied text supplied text. Group LEVELS are screened at their own
    boundary, by ``adapters_fairness._is_missing_level``, which does know the
    string spellings.
    """
    if s is None or (not isinstance(s, str) and _is_nan(s)):
        return "N/A"
    text = str(s)
    return (text[:n] + "…") if len(text) > n else text


def _is_nan(score) -> bool:
    """True when the score is not a usable number (None, NaN, pd.NA, pd.NaT).

    IT DID NOT RETURN A BOOL. ``return score is None or score != score`` looks
    like it is protected by the ``except`` below, and for ``pd.NA`` it is not:
    ``score is None`` is False, so ``or`` evaluates and RETURNS the right
    operand as it is, and ``pd.NA != pd.NA`` is ``pd.NA``. No exception happens
    here, so nothing is caught; the ambiguous value is handed to the CALLER, and
    the caller's ``if`` raises ``TypeError: boolean value of NA is ambiguous``
    from a line that has nothing to do with it. Measured 2026-09-30:
    ``_risk_color(pd.NA)`` raised, from inside the guard whose whole job is to
    keep an unmeasurable score off the colour scale.

    ``bool()`` around the comparison is what makes the ``except`` load-bearing
    instead of decorative. The type annotation was already ``-> bool``.
    """
    try:
        if score is None:
            return True
        return bool(score != score)
    except Exception:
        return True


def _risk_color(score: float) -> str:
    """Map a 0-1 risk score to a text colour.

    Keyed on ``_to_float``, not on ``_is_nan``: a bool passed ``_is_nan``, and
    ``True >= 0.75`` is True, so ``_risk_color(True)`` painted the red HIGH
    colour and ``_risk_label(True)`` printed "HIGH" for a flag that is not a
    risk score at all. A fabricated breach, the mirror of the green-for-nothing
    case the slate branch below exists for. A numeric STRING is now accepted
    instead of raising, which is what these three used to do on a report that
    had been through JSON.
    """
    score = _to_float(score)
    if _is_nan(score):
        return "#64748b"  # slate: unknown, NOT green (NaN fell through to
        # the MINIMAL branch and rendered as a pass)
    if score >= 0.75:
        return "#dc2626"  # red-600
    if score >= 0.50:
        return "#f59e0b"  # amber-500
    if score >= 0.25:
        return "#3b82f6"  # blue-500
    return "#059669"  # emerald-600


def _risk_bg(score: float) -> str:
    """Map a 0-1 risk score to a background colour. See :func:`_risk_color`."""
    score = _to_float(score)
    if _is_nan(score):
        return "#f1f5f9"  # slate tint: unknown
    if score >= 0.75:
        return "#fee2e2"
    if score >= 0.50:
        return "#fef3c7"
    if score >= 0.25:
        return "#dbeafe"
    return "#d1fae5"


def _risk_label(score: float) -> str:
    """Map a 0-1 risk score to a label.

    Delegates to the canonical band definitions (`vfairness._bands`) so the SVG
    badge and the explainer text share one source of truth and cannot drift.

    The COERCION happens first, for the reason in :func:`_risk_color`: a bool is
    an int, ``risk_band(True)`` answered "HIGH", and a numeric string raised
    TypeError inside the bands module. ``_to_float`` returns None for both, and
    ``risk_band`` already renders None as "N/A".
    """
    from .._bands import risk_band

    return risk_band(_to_float(score))


def _severity_icon(level: str) -> str:
    """Map a severity level string to an icon character. Absence is "?", not a tick.

    ``str(None).lower()`` is the string ``"none"``, which is a live KEY of the
    map below, so ``_severity_icon(None)`` returned "✓", the tick this library
    draws for a clean row. Measured 2026-09-30. It is the identical trap
    ``adapters._module_score._enum_score`` carries a comment about ("without it
    'none' is a live key of _enum_map and an absent severity scored 0.05"), one
    module along, and in the louder direction: a tick is the most-read mark on
    a table.

    ``RiskLevel.NONE`` and the literal string "none" still get the tick: those
    are severities a detector actually reported.
    """
    if level is None or (not isinstance(level, str) and _is_nan(level)):
        return "?"
    level = str(level).lower()
    icons = {
        "critical": "✗",
        "high": "!",
        "medium": "⚠",
        "low": "○",
        "none": "✓",
        "negligible": "✓",
        "adequate": "✓",
    }
    return icons.get(level, "?")


# Public API


def render_svg(template_name: str, data: Dict[str, Any], skin: str = "blanco") -> str:
    """
    Render an SVG report from a named template and a data dictionary.

    Parameters
    ----------
    template_name : str
        Template name (without extension), e.g. ``'bias_audit'``.
    data : dict
        Flat or nested dictionary consumed by the template.
    skin : str
        Design scheme applied to the rendered output. Defaults to ``'blanco'``,
        the platform's one design language, so every render (reports, charts,
        gallery) is Blanco. See :mod:`vfairness.rendering.skins`.

    Returns
    -------
    str
        Complete SVG markup.
    """
    # Validate template_name to prevent path traversal
    import re

    if not re.match(r"^[a-zA-Z0-9_]+$", template_name):
        raise ValueError(
            f"Invalid template_name '{template_name}': "
            "must contain only alphanumeric characters and underscores"
        )

    env = _get_jinja_env()
    if env is None:
        raise_jinja2_missing()

    # Build the chart's structured explanation from its data (never raises).
    # This drives (a) the on-canvas explanation, auto-generated when the caller
    # supplied none, and (b) the accessible <title>/<desc>/<metadata> layer.
    from .explain import build_explanation, inject_accessibility

    chart_expl = build_explanation(template_name, data)

    render_data = dict(data)
    # explanation semantics: None (or absent) -> auto-generate; "" -> suppress;
    # a string -> use as given. So render_svg is self-explaining by default.
    if render_data.get("explanation") is None:
        render_data["explanation"] = chart_expl.paragraph()

    # The validant.ai mark (VB-COM-2). Every template <use>s the two symbols
    # defined in _shared_defs.svg, and that include is guarded on this flag, so
    # one value here turns the wordmark and the corner icon off across all of
    # them. setdefault, so a caller can still override a single render.
    render_data.setdefault("branding", branding_enabled())

    template = env.get_template(f"{template_name}.svg")
    try:
        rendered = template.render(**render_data)
    except (ZeroDivisionError, OverflowError, TypeError) as exc:
        # Malformed caller data (stringified numbers compared to ints, None in
        # arithmetic, Infinity through |int, empty lists used as divisors)
        # surfaces here as low-level errors deep inside Jinja. Data is NOT
        # silently coerced or replaced; instead the failure is re-raised as a
        # single clear ValueError naming the template and the root cause, so
        # callers can find and fix the offending field.
        raise ValueError(
            f"template '{template_name}' could not render: "
            f"{type(exc).__name__}: {exc}. The data dict likely contains a "
            "non-numeric value (e.g. a stringified number, None, or Infinity) "
            "where the template does arithmetic or comparison."
        ) from exc

    # Strip characters that are invalid in XML 1.0 (C0 controls except
    # tab/newline/carriage-return). Jinja's autoescape handles & < > but not
    # these; one control character in a user-supplied string made the whole
    # SVG unparseable.
    rendered = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", rendered)

    # Unbranded output must not merely LOOK unbranded. The guard in
    # _shared_defs.svg drops the symbols, which is enough to render nothing,
    # but every template still carries a <use href="#validant-logo"> pointing
    # at the now-absent id. That leaves our name sitting in the markup of a
    # file someone hands to a client or a regulator, so the dangling
    # references come out here too and the word disappears completely.
    if not render_data["branding"]:
        rendered = re.sub(
            r'[ \t]*<use\b[^>]*?(?:xlink:)?href="#validant-[^"]*"[^>]*?/>[ \t]*\n?',
            "",
            rendered,
        )

    # Apply the Blanco skin (recolour + zero-radius + editorial type), then add
    # the accessibility / machine-readable layer as the final step.
    from .skins import apply_skin

    styled = apply_skin(rendered, skin)
    return inject_accessibility(styled, chart_expl)
