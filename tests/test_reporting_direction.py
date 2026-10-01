"""Wave 3 pins for the reporting dashboard adapter's metric direction.

This is the THIRD file in which one bug class has been found: a metric's
better-direction guessed by a SUBSTRING test, which grades a maximal violation
as a pass. Wave 1 fixed the deployment gate, wave 2 fixed the shared helper we
had just written with the same flaw, and acceptance then found it live in
``rendering/adapters_reporting.py``:

    return n.endswith("_ratio") or "disparate impact" in n or "disparate_impact" in n

``disparate_impact_difference`` is a violation MAGNITUDE, but that test matched
the ``disparate_impact`` token inside it, so the row was graded as a four-fifths
ratio. A 0.45 gap against a 0.10 bound therefore rendered ``breached=False`` and
the #059669 GREEN swatch on the canvas an auditor reads.

The fix is not a repaired local test, it is the DELETION of the local test: the
adapter now asks
``evaluation.vfairness_metrics._metric_direction``, which owns the whole
precedence table. Every assertion below is made against the rendered SVG string
wherever the defect was visible there, because the SVG is the artifact that
leaves the building.

Three states, never two. A metric whose direction cannot be resolved, or whose
value was never measured, is could-not-check: it must not be able to reach the
green swatch, the PASS badge, or a GREEN headline status.
"""

from __future__ import annotations

import math
import re

import pytest

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)

from vfairness.evaluation.vfairness_metrics._metric_direction import (  # noqa: E402
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)
from vfairness.rendering.adapters_reporting import (  # noqa: E402
    _is_breached,
    _is_ratio_metric,
    _row_direction,
    reporting_dashboard_to_svg,
)

#: The green swatch the adapter asks for on a passing metric's VALUE cell.
_PASS_GREEN = "#059669"

#: The row STATUS badge, as the template writes it. Read the VERDICT from here
#: and from the breach report, never from the cell colour: ``rendering.skins``
#: deliberately collapses every red and every amber onto ONE terracotta warn
#: hue, so colour can prove "not green" but cannot separate BREACH from WARN.
#:
#: The label charset admits a SPACE, so the could-not-check badge "NOT CHECKED"
#: is read like its three siblings. It was ``[A-Z]+`` while the badge column had
#: only PASS / WARN / BREACH in it, and when the third state arrived this reader
#: returned the empty list for the unchecked row: byte-for-byte the same answer
#: it gives for a canvas with NO rows on it. A verifier that cannot see the
#: third state cannot tell a withheld verdict from a missing one.
#:
#: What is still pinned, deliberately, is the markup CONVENTION
#: (``font-size="8" font-weight="600"``), which every arm of the badge branch in
#: ``reporting_dashboard.svg`` now shares. Loosening that instead would let the
#: next badge drift to its own styling and go invisible here all over again.
_BADGE_RE = re.compile(
    r'text-anchor="middle" font-size="8" font-weight="600" fill="[^"]*">([A-Z][A-Z ]*[A-Z])</text>'
)

#: The badge an ungraded row carries. Not PASS, and not WARN either: see the
#: block comment above ``test_unknown_direction_metric_fails_closed_rather_than_green``.
_UNCHECKED_BADGE = "NOT CHECKED"


def _report(*metrics, status="green", score=90):
    """A dashboard report whose only content under test is the metric rows."""
    return {
        "health_score": {"score": score, "status": status, "components": {}},
        "metrics": list(metrics),
        "tier": "OPERATIONAL",
        "sections": [],
        "alerts": [],
        "recommendations": [],
    }


def _rendered_variants(colour):
    """Every hex the finished SVG may carry for *colour*.

    ``render_svg`` recolours through :mod:`vfairness.rendering.skins`, so the
    adapter's ``#059669`` reaches the page as the skin's pass green. Asserting
    on the adapter's own constant alone would be a proxy for what the reader
    sees, and asserting on the skinned value alone would break if the skin is
    off, so both are accepted.
    """
    variants = {colour.lower()}
    try:
        from vfairness.rendering.skins import DEFAULT_SKIN, apply_skin

        probe = apply_skin(f'<svg><rect fill="{colour}"/></svg>', DEFAULT_SKIN)
        found = re.search(r'fill="(#[0-9a-fA-F]+)"', probe)
        if found:
            variants.add(found.group(1).lower())
    except Exception:  # pragma: no cover - skins is optional to this assertion
        pass
    return variants


def _value_is_green(svg, display_value):
    """True when the row whose VALUE is *display_value* was painted pass-green.

    The template writes ``fill="{{ m.color }}">{{ m.display_value }}<``, so the
    colour and the number it belongs to are read back as one token. Matching a
    bare "#059669" anywhere in the document would be a proxy: the palette uses
    that green elsewhere on the page.
    """
    return any(
        f'fill="{variant}">{display_value}<' in svg.lower()
        for variant in _rendered_variants(_PASS_GREEN)
    )


def _badges(svg):
    """The row STATUS badges, in row order: PASS, WARN, BREACH or NOT CHECKED.

    An EMPTY list is not evidence of anything: it is what a canvas with no
    metric rows returns, and it is what this reader returned for an ungraded row
    while it was blind to the third state. Assert on the CONTENT of the list,
    never on its emptiness.
    """
    return _BADGE_RE.findall(svg)


# 1.  The defect: a disparate-impact DIFFERENCE is not a four-fifths ratio


def test_disparate_impact_difference_at_0_45_is_breached_and_never_green():
    """NEGATIVE case, and the exact input acceptance found rendering green.

    0.45 against a 0.10 bound is a 350 percent breach of a violation magnitude.
    The substring test read the ``disparate_impact`` token, inverted the
    comparison to "at or above the minimum", and painted #059669.
    """
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_difference", "value": 0.45, "threshold": 0.10})
    )
    assert svg, "rendered nothing"
    assert not _value_is_green(svg, "0.4500"), (
        "a 0.45 disparate_impact_difference against a 0.10 bound was painted green"
    )
    assert _badges(svg) == ["BREACH"], "a maximal violation did not carry the BREACH badge"
    assert "THRESHOLD BREACH REPORT" in svg, "the breach is not reported at all"


def test_disparate_impact_difference_below_its_bound_still_passes():
    """Positive control: the fix is a direction fix, not a blanket refusal.

    A small disparate-impact difference is a genuine pass and must stay green,
    otherwise the family has simply been inverted a second time.
    """
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_difference", "value": 0.01, "threshold": 0.10})
    )
    assert _value_is_green(svg, "0.0100")
    assert _badges(svg) == ["PASS"]
    assert "THRESHOLD BREACH REPORT" not in svg


def test_disparate_impact_gap_is_also_a_violation_magnitude():
    """The token, not the word: ``_gap`` names a magnitude just as ``_difference`` does."""
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_gap", "value": 0.45, "threshold": 0.10})
    )
    assert not _value_is_green(svg, "0.4500")
    assert _badges(svg) == ["BREACH"]
    assert "THRESHOLD BREACH REPORT" in svg


# 2.  The genuine ratio family keeps its four-fifths floor


def test_disparate_impact_ratio_at_0_90_against_0_80_renders_fine():
    """0.90 clears the four-fifths rule. Higher is better here, so this is no breach.

    It lands in the dashboard's existing proximity band rather than on the green
    swatch, because 0.90 is only 12.5 percent above the 0.80 floor and the
    adapter warns inside 20 percent. The point under test is the DIRECTION: it
    must not be read as a violation magnitude and reported as a breach.
    """
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_ratio", "value": 0.90, "threshold": 0.80})
    )
    assert "THRESHOLD BREACH REPORT" not in svg, "a passing ratio was reported as a breach"
    assert _badges(svg) == ["WARN"], "0.90 against a 0.80 floor is a pass inside the warn band"
    assert "[UNCHECKED]" not in svg, "a resolvable ratio was treated as could-not-check"


def test_disparate_impact_ratio_clear_of_the_floor_is_green():
    """Positive control on the same family, outside the proximity band."""
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_ratio", "value": 1.00, "threshold": 0.80})
    )
    assert _value_is_green(svg, "1.0000")
    assert _badges(svg) == ["PASS"]
    assert "THRESHOLD BREACH REPORT" not in svg


def test_disparate_impact_ratio_below_the_floor_is_a_breach():
    """NEGATIVE case for the same family: 0.50 fails the four-fifths floor."""
    svg = reporting_dashboard_to_svg(
        _report({"name": "disparate_impact_ratio", "value": 0.50, "threshold": 0.80})
    )
    assert not _value_is_green(svg, "0.5000")
    assert _badges(svg) == ["BREACH"]
    assert "THRESHOLD BREACH REPORT" in svg


def test_bare_disparate_impact_label_is_still_a_ratio():
    """The human label and the snake name both normalise onto the ratio metric."""
    assert _is_ratio_metric("disparate impact") is True
    assert _is_ratio_metric("disparate_impact") is True
    assert _is_ratio_metric("disparate_impact_difference") is False
    assert _is_ratio_metric("calibration_difference") is False


# 3.  Fail closed: unknown direction, contradiction, unmeasurable value

# WHY THE EXPECTED BADGE IN THIS SECTION CHANGED FROM "WARN" TO "NOT CHECKED"
# (2026-08-28). Read this before relaxing any of the four assertions below.
#
# The property these four tests exist to pin is unchanged and is not weakened
# here: an unknown direction, an unmeasurable value and an unmeasurable
# threshold must NEVER render as a pass. What changed is that the old
# expectation pinned the WORKAROUND rather than the property.
#
# ``reporting_dashboard.svg`` used to test ``{% if m.breached %}...{% else %}
# PASS``, two states for three outcomes, so an ungraded row fell through to the
# green PASS badge. The adapter could not add a state from its side, so it sent
# ``warning=True`` for such a row purely to divert it into the WARN arm, and
# ``adapters_reporting`` says so in as many words at the site
# ("The `warning` flag cannot be dropped ... dropping it would badge the row
# PASS"). "WARN" on that row was therefore never a verdict anybody reached: it
# was the name of the diversion. Asserting on it froze the two-state world in
# place, and it is the same shape of defect the whole campaign is about, a row
# that reported nothing being handed a graded label.
#
# The template now carries the real third arm and badges the row NOT CHECKED in
# slate, which is what an auditor sees on the canvas today. Pinning that is
# STRICTLY STRONGER than the old line: "WARN" was satisfied by the collapse,
# whereas "NOT CHECKED" is satisfied only by a distinct third state, and each
# test additionally asserts that no PASS badge is drawn anywhere on the canvas,
# which is the property itself rather than a proxy for it. A revert of the
# template arm now fails these tests instead of passing them.
#
# Corroboration, not judgement: tests/test_desc_agrees_with_canvas.py
# ::test_an_unchecked_dashboard_row_draws_no_verdict_and_no_all_clear asserts
# the complement ("WARN" not in the canvas text, NOT CHECKED twice) and is
# green. The two files could not both be satisfied, and the assertion that had
# to yield is the one describing the defect.


def test_unknown_direction_metric_fails_closed_rather_than_green():
    """A metric the shared helper cannot resolve is could-not-check.

    Before this wave every unresolved name fell through to lower-is-better, so
    a house metric with no known direction was silently graded and could reach
    the green swatch and a PASS badge.
    """
    svg = reporting_dashboard_to_svg(
        _report({"name": "bespoke_house_metric", "value": 0.42, "threshold": 0.10})
    )
    assert not _value_is_green(svg, "0.4200"), "an unchecked metric reached the green swatch"
    assert "PASS" not in _badges(svg), "an unchecked metric carried a PASS badge"
    assert _badges(svg) == [_UNCHECKED_BADGE], "the ungraded row is not badged with the third state"
    assert "[UNCHECKED]" in svg, "the row does not say the threshold was never applied"
    assert "NOT CHECKED" in svg
    assert "THRESHOLD BREACH REPORT" not in svg, "could-not-check was painted as a measured failure"


def test_unchecked_metric_downgrades_the_green_headline_status():
    """A GREEN headline cannot certify a metric that was never compared.

    The health score is computed upstream, so it will happily report GREEN
    beside a row we have just established was not checked.
    """
    svg = reporting_dashboard_to_svg(
        _report(
            {"name": "bespoke_house_metric", "value": 0.42, "threshold": 0.10},
            status="green",
            score=95,
        )
    )
    assert ">GREEN<" not in svg, "the GREEN all-clear survived beside an unchecked metric"
    assert "COULD NOT CHECK" in svg, "the narrative does not state what was not checked"
    assert "bespoke_house_metric" in svg


def test_green_headline_survives_when_everything_was_checked():
    """Positive control for the downgrade: a fully checked report keeps GREEN."""
    svg = reporting_dashboard_to_svg(
        _report(
            {"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10},
            status="green",
            score=95,
        )
    )
    assert ">GREEN<" in svg
    assert "COULD NOT CHECK" not in svg


def test_a_declared_metric_type_may_resolve_but_never_contradict_the_name():
    """A row labelled "ratio" whose NAME is a violation magnitude is contradictory.

    Taking the declaration would re-open exactly the defect this wave closes:
    ``disparate_impact_difference`` graded as a four-fifths ratio. Taking the
    name silently discards what the producer said. Neither is safe, so the
    answer is could-not-check.
    """
    svg = reporting_dashboard_to_svg(
        _report(
            {
                "name": "demographic_parity_difference",
                "value": 0.45,
                "threshold": 0.10,
                "metric_type": "ratio",
            }
        )
    )
    assert not _value_is_green(svg, "0.4500"), "a contradictory row reached the green swatch"
    assert "[UNCHECKED]" in svg
    assert "PASS" not in _badges(svg), "a contradictory row carried a PASS badge"
    assert _badges(svg) == [_UNCHECKED_BADGE]


def test_a_declared_metric_type_can_resolve_a_name_with_no_signal():
    """The declaration is still useful where the helper genuinely cannot tell."""
    from vfairness.evaluation.vfairness_metrics._metric_direction import MetricDirection

    assert metric_direction("bespoke_house_metric") is MetricDirection.UNKNOWN
    assert _row_direction("bespoke_house_metric", "ratio") is MetricDirection.HIGHER_IS_BETTER
    assert _row_direction("bespoke_house_metric", "difference") is MetricDirection.LOWER_IS_BETTER
    # An agreeing declaration changes nothing.
    assert (
        _row_direction("demographic_parity_difference", "difference")
        is MetricDirection.LOWER_IS_BETTER
    )


def test_an_unmeasurable_value_is_not_a_pass():
    """NaN loses every comparison, so ``value > threshold`` reports "not breached"
    for a metric that was never measured. That is a pass rendered from nothing."""
    svg = reporting_dashboard_to_svg(
        _report({"name": "demographic_parity_difference", "value": float("nan"), "threshold": 0.10})
    )
    assert "PASS" not in _badges(svg), "an unmeasured metric was rendered as passing"
    assert _badges(svg) == [_UNCHECKED_BADGE]
    assert "[UNCHECKED]" in svg
    assert not _value_is_green(svg, "nan")


def test_an_unmeasurable_threshold_is_not_a_pass():
    svg = reporting_dashboard_to_svg(
        _report({"name": "demographic_parity_difference", "value": 0.02, "threshold": float("nan")})
    )
    assert "PASS" not in _badges(svg), "an unmeasurable bound was rendered as passing"
    assert _badges(svg) == [_UNCHECKED_BADGE]
    assert "[UNCHECKED]" in svg


def test_a_negative_signed_difference_is_still_a_breach():
    """A violation magnitude is unsigned. ``-0.45`` against a 0.10 bound is a
    0.45 violation, and the signed test reported it as no breach at all."""
    svg = reporting_dashboard_to_svg(
        _report({"name": "demographic_parity_difference", "value": -0.45, "threshold": 0.10})
    )
    assert "THRESHOLD BREACH REPORT" in svg, "a negative violation magnitude read as no breach"
    assert _badges(svg) == ["BREACH"]
    assert not _value_is_green(svg, "-0.4500")


# 3b. The KPI cards: a fabricated BREACH, and the half no narrative can cover
#
# WHY THIS SECTION IS IN THIS FILE. It is not about metric direction, and it
# would sit more naturally beside the scorecard tests in
# tests/test_rowlevel_validator_reporting.py. It is here because the fix had NO
# EXECUTABLE GUARD ANYWHERE and this is the file that ships with it.
#
# Measured 2026-08-28: with the adapter's KPI withholding removed and every
# other file untouched, 122 tests across the whole reporting surface still
# passed. The nearest guard, ::test_unreported_kpi_subscores_are_named_rather_
# than_read_as_failures, asserts only on the NARRATIVE sentence, which is
# produced by a different branch and survives the defect intact. So the three
# red 0/100 cards could be reinstated, deliberately or by a merge, without one
# test going red.
#
# That is exactly how this defect reached wave 15 in the first place: the
# withholding branch was added to reporting_dashboard.svg in one change and the
# adapter kept sending `int(... or 0)` with no flag, so `default(true)` took the
# measured arm and the cards went on painting red zeros for sub-scores nobody
# computed. Two half fixes, no failing test, and the surface unchanged. A guard
# that reads the CARD, not the sentence about the card, is what makes that
# visible.


#: The three KPI card titles, in the order the template draws them.
_KPI_TITLES = ("METRIC COMPLIANCE", "ALERT SCORE", "DRIFT STABILITY")


def _kpi_card(svg, title):
    """The value drawn INSIDE one KPI card, read as the next text node.

    The card's number and its "not measured" replacement are different text
    nodes with different font sizes and fills, so this reads the CONTENT and
    lets the template decide the styling. Reading the number would be a proxy:
    the whole defect is that an absence and a measured 0 rendered identically.
    """
    anchor = f">{title}</text>"
    start = svg.index(anchor) + len(anchor)
    found = re.search(r"<text[^>]*>([^<]*)</text>", svg[start:])
    assert found, f"no value node after the {title} card title"
    return found.group(1)


def _kpi_cards(svg):
    return {title: _kpi_card(svg, title) for title in _KPI_TITLES}


def _report_with_components(components):
    report = _report({"name": "demographic_parity_difference", "value": 0.01, "threshold": 0.10})
    report["health_score"]["components"] = components
    return report


def test_an_unreported_kpi_subscore_is_withheld_from_its_card_not_drawn_as_zero():
    """FABRICATED BREACH. A health record with no ``components`` sent 0 to all
    three cards, and the template colours them ``{% if kpi >= 70 %}``, so an
    absent sub-score rendered a failing 0/100 in red.

    This is the fabricated BREACH direction, not the fabricated all-clear, and
    it is not the safer error: it sends someone after a compliance problem that
    was never measured, on a surface an auditor reads as a compliance finding.
    """
    svg = reporting_dashboard_to_svg(_report_with_components({}))
    assert _kpi_cards(svg) == {title: "not measured" for title in _KPI_TITLES}
    # And it is stated in words too, so the reason survives export to text and
    # reaches a reader who never sees the cards.
    assert "NOT MEASURED: 3 of 3 KPI sub-score(s)" in svg


def test_a_measured_zero_kpi_subscore_is_still_drawn_as_a_failing_zero():
    """POSITIVE CONTROL, and the over-correction this fix must not become.

    0 is a measurement. A sub-score genuinely computed as 0 is a real failing
    score and has to keep its red 0/100; withholding it would turn the
    fabricated breach into a suppressed one, which is the same defect mirrored.
    ``int(... or 0)`` could not tell these two inputs apart at all.
    """
    measured = {"metric_compliance": 0, "alert_frequency": 0, "drift_stability": 0}
    svg = reporting_dashboard_to_svg(_report_with_components(measured))
    assert _kpi_cards(svg) == {title: "0" for title in _KPI_TITLES}
    assert "not measured" not in svg
    assert "NOT MEASURED" not in svg


def test_a_partly_reported_health_record_withholds_only_the_absent_cards():
    """Per card, never all-or-nothing: what WAS measured is still reported."""
    svg = reporting_dashboard_to_svg(_report_with_components({"metric_compliance": 95}))
    assert _kpi_cards(svg) == {
        "METRIC COMPLIANCE": "95",
        "ALERT SCORE": "not measured",
        "DRIFT STABILITY": "not measured",
    }
    assert "NOT MEASURED: 2 of 3 KPI sub-score(s)" in svg


def test_a_fully_reported_health_record_draws_the_cards_unchanged():
    """POSITIVE CONTROL: the healthy canvas must not have moved at all."""
    reported = {"metric_compliance": 95, "alert_frequency": 90, "drift_stability": 88}
    svg = reporting_dashboard_to_svg(_report_with_components(reported))
    assert _kpi_cards(svg) == {
        "METRIC COMPLIANCE": "95",
        "ALERT SCORE": "90",
        "DRIFT STABILITY": "88",
    }
    assert "not measured" not in svg


# 4.  Anti-drift: the row verdict and the shared helper cannot disagree


_AGREEMENT_TABLE = [
    ("disparate_impact_difference", 0.45, 0.10),
    ("disparate_impact_difference", 0.01, 0.10),
    ("disparate_impact_ratio", 0.90, 0.80),
    ("disparate_impact_ratio", 0.50, 0.80),
    ("calibration_difference", 0.90, 0.05),
    ("multicalibration", 0.90, 0.05),
    ("demographic_parity_difference", -0.45, 0.10),
    ("demographic_parity_ratio", 0.00, 0.80),
    ("worst_group_accuracy", 0.55, 0.70),
    ("equal_opportunity_diff", 0.12, 0.10),
]


@pytest.mark.parametrize("name,value,threshold", _AGREEMENT_TABLE)
def test_row_breach_test_agrees_with_the_shared_threshold_check(name, value, threshold):
    """``_is_breached`` is the adapter's arithmetic; ``check_threshold`` is the
    gate's verdict. If they drift apart, the canvas and the gate disagree about
    the same model, which is how a false pass survives review."""
    from vfairness.evaluation.vfairness_metrics._metric_direction import MetricDirection

    direction = metric_direction(name)
    assert direction is not MetricDirection.UNKNOWN, f"table entry {name} lost its direction"
    hib = direction is MetricDirection.HIGHER_IS_BETTER
    outcome, _reason = check_threshold(name, value, threshold)
    assert outcome is not ThresholdOutcome.COULD_NOT_CHECK
    expected = outcome is ThresholdOutcome.FAIL
    assert _is_breached(value, threshold, higher_is_better=hib) is expected, (
        f"{name}={value} vs {threshold}: the row says "
        f"{_is_breached(value, threshold, higher_is_better=hib)}, the gate says {expected}"
    )


def test_the_adapter_holds_no_local_direction_rules_any_more():
    """The reason this bug class returned three times is that each site kept its
    OWN copy of the rules. This module must own none.

    The scan is the WIDENED repo guard from ``test_audit_final_release``, not a
    second bespoke one: a private copy here would go stale exactly the way the
    narrow marker list did, and would then certify this file as clean while the
    same defect sat in it.
    """
    import importlib.util
    import pathlib

    from vfairness.rendering import adapters_reporting

    guard_path = pathlib.Path(__file__).with_name("test_audit_final_release.py")
    spec = importlib.util.spec_from_file_location("_wave3_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    path = pathlib.Path(adapters_reporting.__file__)
    assert not guard._substring_direction_hits(path), (
        "a substring direction test is back in adapters_reporting.py"
    )

    # The module must still EXPLAIN the incident in prose, so the next author
    # does not delete the delegation as unexplained indirection. That erasure is
    # how the cali-BRATIO-n understanding was lost the first time.
    src = path.read_text(encoding="utf-8")
    assert "disparate_impact_difference" in src
    assert "_metric_direction" in src


def test_math_import_guard():
    assert math.isfinite(1.0)


# ───────────────────────────────────────────────────────────────────────────
# N-07: the notice panel must state the absence that ACTUALLY happened.
#
# reporting_dashboard_to_svg raises `not_assessable` for two different
# absences and passes the right sentence in `not_assessable_reason`. The panel
# hard-coded the first one, so a report that WAS supplied but carried no health
# score got a canvas that contradicted itself four lines apart: the subtitle
# said "No health score was reported: the scorecard is withheld, not zero" and
# the panel underneath said "No report was supplied".
#
# This was recorded CLOSED in the row-level register and was live. The closure
# note quoted text that IS on the canvas, but from the NARRATIVE panel roughly
# 300px lower, which is fed by a different branch and was correct all along.
# Nothing read the notice panel itself: with the defect reinstated, 1038 tests
# passed. So these read the panel, addressed by its own y band, not the page.
# ───────────────────────────────────────────────────────────────────────────

_NOTICE_BAND = 260  # header 140 + the notice panel; the narrative sits far below


#: The smallest height a full-width rect must have to count as a PANEL.
#: ``reporting_dashboard.svg`` also draws 1.5px full-width RULES (the breach
#: report's red hairline sits on the breach panel's own top edge, at the same y
#: as the panel). Counting a rule as a panel makes a stacking walk measure the
#: next gap from the rule's bottom, 1.5px below the panel top, and report a
#: ~90px false gap on a correct canvas. The smallest real panel here is 78px, so
#: this threshold separates the two by an order of magnitude.
_MIN_PANEL_H = 8


def _full_width_panels(svg):
    """Every full-width panel as ``(top, height)``, unsorted (draw order).

    The markup is NOT in top-to-bottom order: the not_assessable branch emits
    the breach panel before the narrative panel regardless of where each one
    lands, so any caller comparing neighbours has to sort first.
    """
    return [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<rect x="36" y="([0-9.]+)" width="644" height="([0-9.]+)"', svg)
        if float(m.group(2)) >= _MIN_PANEL_H
    ]


def _canvas_height(svg):
    """The height of the canvas an auditor actually sees.

    Read from the viewBox, not from ``total_h`` in the template: the viewBox is
    what clips, and the whole N-08 hazard is a panel drawn outside it.
    """
    box = re.search(r'viewBox="0 0 [0-9.]+ ([0-9.]+)"', svg)
    assert box, "no viewBox on the rendered canvas"
    return float(box.group(1))


def _notice_lines(svg):
    """Text drawn inside the notice panel only, so the narrative panel further
    down the canvas cannot satisfy an assertion about this one."""
    out = []
    for m in re.finditer(r'<text[^>]*\by="([-0-9.]+)"[^>]*>(.*?)</text>', svg, re.S):
        if float(m.group(1)) <= _NOTICE_BAND:
            out.append(" ".join(m.group(2).split()))
    return out


def test_a_report_with_no_health_score_is_not_called_a_missing_report():
    svg = reporting_dashboard_to_svg({"model_name": "m"})
    panel = " ".join(_notice_lines(svg))

    assert "No report was supplied" not in panel, (
        "the notice panel says no report was supplied, but one WAS supplied and "
        f"is described correctly in the subtitle. Panel text: {panel!r}"
    )
    assert "carries no health score" in panel, (
        f"the notice panel does not state the absence that happened: {panel!r}"
    )


def test_a_genuinely_absent_report_still_says_so():
    """CONTROL. The other branch must keep its own, different, true sentence."""
    svg = reporting_dashboard_to_svg(None)
    panel = " ".join(_notice_lines(svg))

    assert "No report was supplied" in panel, panel
    assert "carries no health score" not in panel, panel


def test_the_notice_panel_grows_to_fit_its_reason_instead_of_clipping():
    """The two reasons are 122 and 208 characters. A fixed 96px panel drew the
    longer one off its own bottom edge, which is a silent way to lose the half
    of the sentence that names what was missing."""

    def panel_height(svg):
        m = re.search(r'<rect x="36" y="140" width="644" height="([0-9.]+)"', svg)
        assert m, "could not find the notice panel rect"
        return float(m.group(1))

    short = reporting_dashboard_to_svg(None)  # 122-char reason, 2 lines
    long = reporting_dashboard_to_svg({"model_name": "m"})  # 208-char, 3 lines

    # The invariant is that the panel is DERIVED from its content, not that the
    # current strings happen to fit. A fixed 96px box holds three lines only by
    # 4px of luck, so an assertion that the last baseline is inside the box
    # passes on the broken version too. Measured that: it did.
    assert panel_height(long) > panel_height(short), (
        f"the notice panel is {panel_height(long)}px for a 3-line reason and "
        f"{panel_height(short)}px for a 2-line one, so its height is fixed rather "
        "than computed. The next reason longer than these two is drawn outside "
        "its own box, and the half that names what was missing is the half that "
        "disappears."
    )
    # And every line of the longer reason is actually drawn, not just the first two.
    assert "no KPI sub-score was reported" in " ".join(_notice_lines(long))


def test_a_not_assessable_canvas_with_a_breach_row_does_not_stack_it_on_the_notice():
    """N-08, pinned at the TEMPLATE level because the adapter cannot reach it.

    The not_assessable branch rebases health_y, nlg_y, rec_y, exp_y and total_h
    onto the notice panel, and used to leave breaches_y and alerts_y derived
    from the collapsed metric grid. It was latent only because the adapter
    passes both lists empty and the panels self-suppress, so no adapter-level
    test can reach it: a defect nothing can trigger today is still a defect
    waiting for the first caller who sets both.
    """
    from vfairness.rendering.engine import render_svg

    svg = render_svg(
        "reporting_dashboard",
        {
            "title": "T",
            "subtitle": "s",
            "timestamp": "2026-08-28 10:00",
            "explanation": None,
            "model_name": "m",
            "model_version": "1",
            "not_assessable": True,
            "not_assessable_reason": "The report supplied carries no health score.",
            "status_color": "#94a3b8",
            "status_bg": "#f1f5f9",
            "metrics": [],
            "breaches": [
                {
                    "metric_name": "demographic_parity_difference",
                    "display_value": "0.4500",
                    "display_threshold": "0.10",
                    "breach_pct": 350,
                    "bar_w": 80,
                    "severity": "HIGH",
                    "affected_groups": "A, B",
                    "insight": "d",
                }
            ],
            "alerts": [],
            "recommendations": [],
            "narrative": "",
        },
    )

    notice = re.search(r'<rect x="36" y="140" width="644" height="([0-9.]+)"', svg)
    assert notice, "no notice panel"
    notice_bottom = 140 + float(notice.group(1))

    canvas_h = _canvas_height(svg)
    panels = sorted(p for p in _full_width_panels(svg) if p[0] > 140)
    assert panels, "no panel below the notice at all"

    # The invariant is SEQUENTIAL STACKING, which is what this branch promises
    # by rebasing every other offset onto the notice.
    #
    # WHY THIS WALKS EVERY PANEL AND ALSO CHECKS THE BOUNDS (2026-08-28). Both
    # halves are here because each one alone was measured to pass on a broken
    # template, and neither observation cancels the other:
    #
    #  * Bounds alone are not enough. With ONLY breaches_y left un-rebased the
    #    panel does not overlap the notice and does not fall off the canvas
    #    (total_h is rebased, so the canvas grows with it); it opens a 208px
    #    dead gap instead. An overlap assertion and a bounds assertion both
    #    passed on that version. That is why the gap is measured at all.
    #  * The gap on the FIRST panel alone is not enough either. Rebase nlg_y
    #    but leave breaches_y and alerts_y on the collapsed metric grid, and
    #    the narrative panel lands 16px under the notice while the breach panel
    #    is drawn at y=442 on a canvas whose viewBox is 412 high: silently
    #    clipped off the bottom, which is the exact hazard N-08 names. Measured
    #    2026-08-28: 35 of 35 tests in this file passed on that template while
    #    `min(tops)` read the innocent panel and reported a 16px gap.
    #
    # So: every consecutive pair, and containment of the last one.
    previous_bottom = notice_bottom
    for top, height in panels:
        gap = top - previous_bottom
        assert 0 <= gap <= 32, (
            f"a panel starts {gap}px after the one above it (top={top}, "
            f"panels below the notice: {panels}). The branch rebases health_y, "
            "nlg_y, rec_y, exp_y and total_h onto the notice; any offset it "
            "forgets is still derived from the collapsed metric grid, and the "
            "reader gets a screen of white space in the middle of a canvas "
            "whose whole message is that nothing was assessed."
        )
        previous_bottom = top + height

    # max(), not the last panel's bottom: the gap walk above would already have
    # caught a panel tall enough to swallow its successor, but the containment
    # claim is about EVERY panel and should not depend on that.
    stack_bottom = max(top + height for top, height in panels)
    assert stack_bottom <= canvas_h, (
        f"the stack ends at {stack_bottom}px on a canvas {canvas_h}px high, so "
        f"a panel is drawn off the bottom edge and silently clipped. Panels below "
        f"the notice: {panels}. A non-empty breach or alert row that the "
        "not_assessable branch forgot to rebase lands here, and the reader is "
        "never told the row exists."
    )
