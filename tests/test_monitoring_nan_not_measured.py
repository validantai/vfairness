"""A drift score that could not be measured must not render as the calmest state.

Finding #11 (fourth-iteration register). A NaN MMD distribution-shift score
rendered on the live monitoring dashboard as a FULL-WIDTH GREEN bar with the
verdict chip STABLE: could-not-check collapsed into pass, on the surface an
operator watches to decide whether a deployed model has drifted.

Three independent halves produced it, and all three had to be fixed:

1. ``_drift_color(nan)``: every ``>=`` against NaN is False, so it fell through
   to the emerald return.
2. ``max(0.0, min(1.0, nan))`` is 1.0 in CPython (``nan < 1.0`` is False, so
   ``min`` returns 1.0), giving ``bar_w = max(4, int(1.0 * 400)) = 400``.
3. The template decided the chip itself with ``{% if s.score >= 0.1 %}``, and
   NaN fails that test exactly as a tiny score does, so the ``{% else %}``
   branch printed STABLE.

Fixing only one half would have left the other two still lying. The pins below
therefore assert the rendered SVG, not just the helpers.

The colour assertions use the RENDERED hexes, not the source constants: the
render engine applies a palette transform, so ``#059669`` in the adapter reaches
the SVG as ``#41ba1b``. Asserting the source constant would be a proxy that
passes while the operator still sees green.
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from vfairness.operations.monitoring.tracker import WindowMetrics
from vfairness.rendering.adapters_monitoring import (
    NOT_MEASURED_COLOR,
    NOT_MEASURED_LABEL,
    _clamp01,
    _drift_color,
    _trend_slope_color,
    monitoring_dashboard_to_svg,
)

# As rendered, after the engine's palette transform.
RENDERED_GREEN = "#41ba1b"
RENDERED_NEUTRAL = "#93a0ab"
TRACK_WIDTH = "280"  # the full-width geometry the defect produced

NAN = float("nan")


def _render(mmd_scores: dict) -> str:
    wm = WindowMetrics(
        batch_id="b1",
        timestamp=datetime(2026, 8, 27, 12, 0, 0),
        sample_count=500,
        metrics={"demographic_parity_difference_gender": 0.02},
        group_rates={"gender": {"f": 0.41, "m": 0.43}},
        alerts={"demographic_parity_difference_gender": False},
        mmd_scores=mmd_scores,
    )
    return monitoring_dashboard_to_svg(wm)


def _mmd_bar(svg: str) -> str:
    """The coloured fill rect of the single MMD row (not the grey track)."""
    bars = re.findall(r'<rect x="220" y="\d+" width="(\d+)" height="10" fill="(#[0-9a-f]{6})"', svg)
    fills = [b for b in bars if b[1] != "#f4f3f2"]
    assert len(fills) == 1, f"expected exactly one MMD fill bar, got {bars}"
    return fills[0]


# --- the defect -------------------------------------------------------------


def test_unmeasurable_drift_score_is_not_labelled_stable():
    svg = _render({"gender": NAN})
    assert NOT_MEASURED_LABEL in svg
    assert "STABLE" not in svg, "an unmeasurable drift score must not read as STABLE"
    assert "SHIFT" not in svg, "nor as a finding; it is could-not-check"


def test_unmeasurable_drift_score_is_not_painted_green():
    svg = _render({"gender": NAN})
    width, color = _mmd_bar(svg)
    assert color != RENDERED_GREEN, "the reassuring colour must not survive a NaN score"
    assert color == RENDERED_NEUTRAL


def test_unmeasurable_drift_score_does_not_draw_a_full_width_bar():
    svg = _render({"gender": NAN})
    width, _ = _mmd_bar(svg)
    assert width != TRACK_WIDTH, (
        "NaN drew the full-width bar because max(0.0, min(1.0, nan)) is 1.0 in CPython"
    )
    assert width == "4", "zero-length clamps to the 4px minimum"


def test_unmeasurable_score_is_not_printed_as_the_string_nan():
    svg = _render({"gender": NAN})
    assert ">nan<" not in svg
    assert ">n/a<" in svg


# --- controls: measurable scores must be completely unaffected ---------------


def test_low_score_still_renders_stable_and_green():
    """Control. The fix must not grey out every row."""
    svg = _render({"gender": 0.01})
    assert "STABLE" in svg
    assert NOT_MEASURED_LABEL not in svg
    width, color = _mmd_bar(svg)
    assert color == RENDERED_GREEN, "a genuinely stable score keeps its green"


def test_high_score_still_renders_shift():
    """Control. A real distribution shift must still be reported as one."""
    svg = _render({"gender": 0.72})
    assert "SHIFT" in svg
    assert NOT_MEASURED_LABEL not in svg
    assert "STABLE" not in svg


def test_measurable_and_unmeasurable_rows_coexist():
    """The NaN row must not suppress or contaminate a real finding beside it."""
    svg = _render({"age": 0.72, "gender": NAN})
    assert "SHIFT" in svg, "the measurable row still reports its shift"
    assert NOT_MEASURED_LABEL in svg, "the unmeasurable row is named as such"
    assert "STABLE" not in svg


# --- the helpers, unit level -------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (NAN, NOT_MEASURED_COLOR),
        (None, NOT_MEASURED_COLOR),
        (0.7, "#dc2626"),
        (0.4, "#f59e0b"),
        (0.1, "#059669"),
    ],
)
def test_drift_color_bands(score, expected):
    assert _drift_color(score) == expected


@pytest.mark.parametrize(
    "slope,expected",
    [(NAN, NOT_MEASURED_COLOR), (0.02, "#dc2626"), (0.007, "#f59e0b"), (0.001, "#059669")],
)
def test_trend_slope_color_bands(slope, expected):
    """The sibling helper had the identical hole: abs(nan) is nan, so it fell to green."""
    assert _trend_slope_color(slope) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(NAN, 0.0), (None, 0.0), (-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_clamp01(value, expected):
    assert _clamp01(value) == expected


def test_clamp01_is_what_the_stdlib_gets_wrong():
    """Pins the exact CPython behaviour that caused the full-width bar."""
    assert max(0.0, min(1.0, NAN)) == 1.0, "the stdlib expression the defect used"
    assert _clamp01(NAN) == 0.0, "the guarded replacement"
