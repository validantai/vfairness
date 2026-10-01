"""Release-pass pins: an SVG must never certify what was never measured.

Three defects, one class. An SVG is an export format: it leaves the building,
gets pasted into a deck, attached to an audit file, and outlives the version
that produced it. So every fairness chart must carry THREE states on its own
canvas (assessed-pass, assessed-fail, could-not-check) and must never collapse
the third into either of the other two.

1. The renderers never read ``assessment["assessable"]``. A report whose own
   assessment block says NOT ASSESSABLE (the only minority group dropped by
   ``min_group_size``, so every disparity metric is vacuous) rendered as green
   FAIR rows, "PASSED 5 / 5", "FAIR RATE 100%", an embedded explanation reading
   "5 of 5 fairness metric(s) passed", and a detailed report reading "Overall
   fairness is FAIR with a score of 100/100". The excluded group's name appeared
   nowhere at all.

2. ``passed = abs(value) <= threshold if threshold else True``. A threshold of
   ``0.0``, a zero-tolerance policy, is FALSY, so the branch returned True and
   the metric rendered green. So did a metric with no configured threshold.

3. The confidence-interval adapter read a ``confidence_intervals`` key with
   ``lower``/``upper`` that nothing in the library has ever emitted (the real
   bootstrap output lives under ``metrics_with_ci`` with
   ``lower_bound``/``upper_bound``), so on genuine output every lookup missed and
   it SYNTHESISED ``value +/- 0.02`` under a heading reading "Confidence
   Intervals". A real band of [0.120, 0.300] drew as [0.190, 0.230], and the
   significance badge read "4 significant" on data the library's own bootstrap
   called not significant.

Every assertion below is made against the REAL rendered SVG string, not against
an adapter's intermediate dict, because the SVG is the artifact that ships.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from vfairness._bands import risk_band  # noqa: F401  (import-time sanity)

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)

from vfairness.evaluation.vfairness_metrics.report import (  # noqa: E402
    classification_fairness_report,
)
from vfairness.rendering import adapters, adapters_fairness  # noqa: E402

# The seven fairness adapters that consume a report dict, by chart name.
FAIRNESS_ADAPTERS = {
    "metrics_bar_chart": adapters_fairness.metrics_bar_chart_to_svg,
    "radar_chart": adapters_fairness.radar_chart_to_svg,
    "group_comparison": adapters_fairness.group_comparison_to_svg,
    "disparity_heatmap": adapters_fairness.disparity_heatmap_to_svg,
    "effect_sizes": adapters_fairness.effect_sizes_to_svg,
    "confidence_intervals": adapters_fairness.confidence_intervals_to_svg,
    "fairness_report": adapters.fairness_report_to_svg,
}

# Marks that mean "this metric passed" on one of these canvases.
_GREEN_PASS_MARKS = ("✓ FAIR", ">FAIR<", "✓ Fair", "✓ Low", "✓ No significant disparity")
# The pass greens as they appear in the FINISHED file. The templates are written
# in emerald (#059669, #d1fae5, ...), but render_svg applies the Blanco skin as
# its last step, so not one of those hexes survives into the SVG that ships.
# Pinning the template-side hex therefore asserted nothing at all: a chart could
# paint every badge green and still satisfy it. These are the post-skin values,
# and test_the_pass_green_pin_is_not_vacuous below keeps them honest.
_PASS_GREEN_MARKS = ("#41ba1b", "#1c6d00", "#e4f3da", "#d6ecca")

# The validant mark's own gradient contains one of those greens, and every
# template embeds it, so the scan runs over the chart body only.
_DEFS_RE = re.compile(r"<defs>.*?</defs>", re.S)

# effect_sizes draws a STATIC x-axis band key: a 20%-opacity tint and the word
# NEGLIGIBLE labelling the |d| < 0.2 zone of the SCALE. It is part of the axis,
# not a verdict about the data, and it is drawn whether or not any lollipop
# exists. The exemption is bounded by
# test_unmeasured_effect_chart_draws_no_data_mark below: the scale key may stay,
# a data mark may not.
_AXIS_KEY_ONLY = {"effect_sizes"}


def _chart_body(svg):
    return _DEFS_RE.sub("", svg)


def _not_assessable_report():
    """A REAL engine report on data where only one group survives filtering.

    200 majority rows and 5 minority rows against ``min_group_size=30``: the
    minority group is dropped from every disparity metric, so no between-group
    comparison happens at all. The minority is also treated far worse, so a
    "FAIR" verdict here is not merely vacuous, it is the opposite of the truth.
    """
    rng = np.random.default_rng(7)
    groups = np.array(["majority"] * 200 + ["minority"] * 5)
    y_true = rng.integers(0, 2, size=205)
    y_pred = y_true.copy()
    y_true[200:] = 1
    y_pred[200:] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, groups, min_group_size=30)


def _assessable_report():
    """A REAL engine report with two well-populated, near-identical groups."""
    rng = np.random.default_rng(5)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, groups, min_group_size=30)


# ── 1. the not-assessable report must not certify anything ──────────────────


def test_the_fixture_really_is_not_assessable():
    """Guard the guard: if the engine ever calls this data assessable, every
    assertion below becomes vacuously true and this file stops protecting
    anything."""
    report = _not_assessable_report()
    assert report["assessment"]["assessable"] is False
    assert report["assessment"]["fairness_score"] is None
    assert "minority" in report["data_info"]["invalid_groups"]


def test_the_pass_green_pin_is_not_vacuous():
    """Guard the guard. A colour pin that no rendered file can ever contain is
    not a pin, and this one was exactly that until the skin was accounted for.
    A chart that DID measure a pass must wear at least one of these greens,
    otherwise the assertion below is testing nothing."""
    svg = _chart_body(adapters_fairness.metrics_bar_chart_to_svg(_assessable_report()))
    assert "✓ FAIR" in svg, "the fixture no longer renders a measured pass"
    assert any(green in svg for green in _PASS_GREEN_MARKS), (
        "no pinned green appears on a passing chart, so the palette has moved "
        "and every 'no green' assertion in this file has gone vacuous"
    )


def test_unmeasured_effect_chart_draws_no_data_mark():
    """The bound on the effect_sizes exemption above: its green axis key is
    allowed to stay only because no DATA is drawn against it. A lollipop, or a
    banded MAX EFFECT tile, would be a verdict."""
    svg = _chart_body(adapters_fairness.effect_sizes_to_svg(_not_assessable_report()))
    assert "<circle" not in svg, "a lollipop was plotted for an effect that was never computed"
    assert "NOT ASSESSABLE" in svg
    for word in ("Negligible", "Small", "Medium", "Large"):
        assert f">{word}<" not in svg, f"an effect was banded as {word} with nothing measured"


@pytest.mark.parametrize("chart", sorted(FAIRNESS_ADAPTERS))
def test_not_assessable_report_renders_no_pass_mark(chart):
    svg = FAIRNESS_ADAPTERS[chart](_not_assessable_report())
    assert svg, f"{chart}: rendered nothing"
    for mark in _GREEN_PASS_MARKS:
        assert mark not in svg, f"{chart}: rendered a pass mark {mark!r} for an unmeasured run"
    if chart in _AXIS_KEY_ONLY:
        return
    for green in _PASS_GREEN_MARKS:
        assert green not in _chart_body(svg), (
            f"{chart}: painted the measured-pass green {green} on a chart that measured nothing"
        )


@pytest.mark.parametrize("chart", sorted(FAIRNESS_ADAPTERS))
def test_not_assessable_report_reports_no_fair_rate_or_score(chart):
    svg = FAIRNESS_ADAPTERS[chart](_not_assessable_report())
    assert "100%" not in svg, f"{chart}: reported a 100% fair rate for an unmeasured run"
    assert "FAIR RATE" not in svg, (
        f"{chart}: kept the FAIR RATE tile; it must be SUPPRESSED, not zeroed"
    )
    assert "PASSED" not in svg, f"{chart}: kept the PASSED tile; it must be SUPPRESSED, not zeroed"
    assert not re.search(r"\b5\s*/\s*5\b", svg), f"{chart}: rendered 'PASSED 5 / 5'"
    assert "100/100" not in svg and ">100<" not in svg, f"{chart}: rendered a 100 score"


@pytest.mark.parametrize("chart", sorted(FAIRNESS_ADAPTERS))
def test_not_assessable_report_says_so_on_canvas(chart):
    svg = FAIRNESS_ADAPTERS[chart](_not_assessable_report())
    assert "NOT ASSESSABLE" in svg or "COULD NOT CHECK" in svg, (
        f"{chart}: no could-not-check signal anywhere on the canvas"
    )


@pytest.mark.parametrize("chart", sorted(FAIRNESS_ADAPTERS))
def test_not_assessable_report_names_the_excluded_group(chart):
    """The dropped group is precisely the one the chart says nothing about, so
    its name must be visible to whoever reads the exported file."""
    svg = FAIRNESS_ADAPTERS[chart](_not_assessable_report())
    assert "minority" in svg, f"{chart}: the excluded group is not named anywhere"


@pytest.mark.parametrize("chart", sorted(FAIRNESS_ADAPTERS))
def test_not_assessable_prose_makes_no_pass_claim(chart):
    """The embedded explanation is the only full sentence on the canvas, so it is
    the sentence a reader believes. It said "5 of 5 fairness metric(s) passed"."""
    svg = FAIRNESS_ADAPTERS[chart](_not_assessable_report())
    desc = re.search(r"<desc>([^<]*)</desc>", svg)
    assert desc, f"{chart}: no accessible <desc>"
    text = desc.group(1)
    assert "COULD NOT CHECK" in text, f"{chart}: <desc> does not say it could not check: {text!r}"
    assert not re.search(r"\d+ of \d+ fairness metric\(s\) passed", text), (
        f"{chart}: <desc> claims metrics passed: {text!r}"
    )
    assert "is FAIR" not in text, f"{chart}: <desc> declares the model fair: {text!r}"


# ── positive control: a real, assessable run still reports its verdict ───────


def test_assessable_report_still_shows_its_pass_verdict():
    """The fix must not turn every chart into could-not-check. A genuinely
    measured, genuinely fair run keeps its green verdict and its fair rate."""
    report = _assessable_report()
    assert report["assessment"]["assessable"] is True
    svg = adapters_fairness.metrics_bar_chart_to_svg(report)
    assert "✓ FAIR" in svg
    assert "PASSED" in svg and "FAIR RATE" in svg
    assert "NOT ASSESSABLE" not in svg
    dash = adapters.fairness_report_to_svg(report)
    assert "NOT ASSESSABLE" not in dash
    assert "FAIRNESS SCORE" in dash


# ── 2. thresholds: 0.0 is a policy, not an absence ──────────────────────────


def _threshold_report(value, threshold, key="demographic_parity_difference"):
    return {
        "metrics": {key: value},
        "thresholds_used": {key: threshold},
        "assessment": {"assessable": True, "fairness_score": 0.0, "summary": ""},
        "group_stats": {
            "a": {"positive_rate": 0.8, "size": 100},
            "b": {"positive_rate": 0.2, "size": 100},
        },
    }


def test_zero_threshold_is_a_policy_not_an_absence():
    """NEGATIVE case. A zero-tolerance threshold must REFUSE a breaching value.

    ``if threshold`` is False for 0.0, so the old code skipped the comparison
    entirely and rendered green. 0.25 against a 0.0 bound is a breach.
    """
    svg = adapters_fairness.metrics_bar_chart_to_svg(_threshold_report(0.25, 0.0))
    assert "✗ FAIL" in svg
    assert "✓ FAIR" not in svg
    assert "100%" not in svg

    cards = adapters._metric_cards(_threshold_report(0.25, 0.0))
    assert [c["state"] for c in cards] == ["fail"]


def test_zero_threshold_still_passes_a_value_that_meets_it():
    """The fix is a real comparison, not a blanket refusal: exactly 0.0 against a
    0.0 bound is within the policy and must still read as a pass."""
    svg = adapters_fairness.metrics_bar_chart_to_svg(_threshold_report(0.0, 0.0))
    assert "✓ FAIR" in svg
    assert "✗ FAIL" not in svg


def test_no_configured_threshold_is_could_not_check_never_fair():
    """A metric with no bound was never checked against anything. That is the
    third state, not a pass."""
    report = {
        # A metric name outside every default-threshold table.
        "metrics": {"bespoke_house_metric": 0.42},
        "thresholds_used": {},
        "assessment": {"assessable": True, "fairness_score": 0.0, "summary": ""},
    }
    svg = adapters_fairness.metrics_bar_chart_to_svg(report)
    assert "✓ FAIR" not in svg, "an unchecked metric rendered as fair"
    assert "NO DATA" in svg
    assert "COULD NOT CHECK" in svg

    cards = adapters._metric_cards(report)
    assert [c["state"] for c in cards] == ["could_not_check"]
    assert cards[0]["passed"] is False


def test_zero_threshold_svg_agrees_with_the_engine_verdict():
    """End to end against the real engine with every threshold set to 0.0: the
    set of metrics the SVG paints green must equal the set the engine passed."""
    rng = np.random.default_rng(11)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    y_pred[(groups == "b") & (rng.random(400) < 0.5)] = 0
    zero = dict.fromkeys(
        [
            "demographic_parity_difference",
            "demographic_parity_ratio",
            "equalized_odds_difference",
            "equal_opportunity_difference",
            "predictive_parity_difference",
        ],
        0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups, thresholds=zero)

    engine_pass = {e["metric"] for e in report["assessment"]["passed_metrics"]}
    cards = {c["key"]: c for c in adapters._metric_cards(report)}
    svg_pass = {k for k, c in cards.items() if c["state"] == "pass"}
    # The chart may never paint green anything the engine failed. It MAY be
    # stricter than the engine, but only by moving a metric into the third
    # state, never by inventing a failure.
    assert svg_pass <= engine_pass, (
        f"the chart passed metrics the engine did not: {sorted(svg_pass - engine_pass)}"
    )
    for key in engine_pass - svg_pass:
        assert cards[key]["state"] == "could_not_check", (
            f"{key}: the chart contradicts the engine's pass with a {cards[key]['state']}"
        )
        assert cards[key]["reason"], f"{key}: downgraded without saying why on the canvas"

    # THE TWO NOW AGREE, which is what this block was waiting for.
    #
    # demographic_parity_ratio is HIGHER-is-better, so a threshold of 0.0 is a
    # required MINIMUM that no value can fall below: the bound cannot be
    # breached, so it grades nothing. Until 2026-09-10 the canvas refused it as
    # could-not-check while the engine's check_threshold still answered PASS
    # (0.49 >= 0.0) and wrote it into passed_metrics, and this test asserted the
    # disagreement, with a note saying it "should now assert full agreement
    # instead" once the engine gained the degenerate-threshold guard.
    #
    # READINESS-6 gave it that guard: report._threshold_entry delegates the
    # could-not-check question to check_threshold rather than deciding it
    # locally, so an unenforceable bound is now refused on BOTH surfaces.
    # Measured: engine_pass and svg_pass are both {predictive_parity_difference},
    # and demographic_parity_ratio is could_not_check on the canvas and absent
    # from passed_metrics.
    assert "demographic_parity_ratio" not in engine_pass, (
        "the engine passed a metric under a bound it cannot breach; the "
        "degenerate-threshold guard in report._threshold_entry has regressed"
    )
    assert cards["demographic_parity_ratio"]["state"] == "could_not_check"
    assert svg_pass == engine_pass, (
        f"the canvas and the engine disagree: canvas {sorted(svg_pass)} vs "
        f"engine {sorted(engine_pass)}"
    )

    # And the fair rate on the canvas is a fraction of the metrics that were
    # actually CHECKED, so it is reported over that denominator rather than
    # borrowing the engine's score, which counted the vacuous pass.
    svg = adapters_fairness.metrics_bar_chart_to_svg(report)
    n_checked = sum(1 for c in cards.values() if c["state"] in ("pass", "fail"))
    n_passed = len(svg_pass)
    assert f"{n_passed} / {n_checked}" in svg
    assert f"{int(round(100 * n_passed / n_checked))}%" in svg
    assert "100%" not in svg


# ── 3. confidence intervals: computed or nothing ────────────────────────────


def _ci_report(n_bootstrap=400):
    rng = np.random.default_rng(3)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    flip = rng.random(400) < 0.35
    y_pred[flip & (groups == "b")] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(
            y_true,
            y_pred,
            groups,
            min_group_size=30,
            include_ci=True,
            n_bootstrap=n_bootstrap,
            random_state=1,
        )


def test_forest_plot_shows_the_real_bootstrap_band():
    """The engine's own ``metrics_with_ci`` bounds must reach the canvas."""
    report = _ci_report()
    assert "confidence_intervals" not in report, (
        "the key the adapter used to read is still absent from real output"
    )
    ci = report["metrics_with_ci"]["demographic_parity_difference"]
    svg = adapters_fairness.confidence_intervals_to_svg(report)

    shown = f"[{ci['lower_bound']:.3f}, {ci['upper_bound']:.3f}]"
    assert shown in svg, f"the computed band {shown} is not on the canvas"

    # And the fabricated band is gone. This is the exact synthesis that used to
    # replace it: value +/- 0.02, dressed as a confidence interval.
    point = ci["point_estimate"]
    fake = f"[{point - 0.02:.3f}, {point + 0.02:.3f}]"
    assert fake not in svg, f"a synthesised band {fake} is still being drawn"


def test_forest_plot_significance_matches_the_bootstrap():
    """The badge counted fabricated bands. It must now count only computed ones,
    and agree with the engine's own significance verdict."""
    report = _ci_report()
    expected = sum(
        1 for ci in report["metrics_with_ci"].values() if ci.get("is_significant") is True
    )
    svg = adapters_fairness.confidence_intervals_to_svg(report)
    n_ci = len(report["metrics_with_ci"])
    if expected:
        assert f"{expected} significant" in svg
    else:
        assert "No significant disparity" in svg
    assert f"{expected} / {n_ci}" in svg, (
        "the SIGNIFICANT tile does not report computed intervals only"
    )


def test_metric_without_a_computed_interval_says_so():
    """A point estimate with no interval keeps its diamond and gains an explicit
    label. It never gains an invented whisker."""
    report = _ci_report()
    # predictive_parity_difference is in `metrics` but has no bootstrap CI.
    assert "predictive_parity_difference" not in report["metrics_with_ci"]
    svg = adapters_fairness.confidence_intervals_to_svg(report)
    assert "(CI not computed)" in svg
    point = report["metrics"]["predictive_parity_difference"]
    fake = f"[{point - 0.02:.3f}, {point + 0.02:.3f}]"
    assert fake not in svg


def test_report_with_no_intervals_suppresses_the_significance_badge():
    """No computed interval means no significance claim. Not "0 significant",
    which reads as a measured all-clear."""
    rng = np.random.default_rng(9)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    y_pred[(groups == "b") & (rng.random(400) < 0.4)] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups, include_ci=False)
    assert "metrics_with_ci" not in report

    svg = adapters_fairness.confidence_intervals_to_svg(report)
    assert "CI not computed" in svg
    assert "significant" not in svg, "a significance claim was made without any interval"
    assert "SIGNIFICANT" not in svg, "the SIGNIFICANT tile survived with nothing to count"
    assert "No significant disparity" not in svg
    # Every row is labelled, and not one carries a fabricated band.
    for key, value in report["metrics"].items():
        fake = f"[{value - 0.02:.3f}, {value + 0.02:.3f}]"
        assert fake not in svg, f"{key}: synthesised band still drawn"


def test_forest_plot_prose_makes_no_significance_claim_without_intervals():
    rng = np.random.default_rng(9)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, size=400)
    y_pred = y_true.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_fairness_report(y_true, y_pred, groups, include_ci=False)
    svg = adapters_fairness.confidence_intervals_to_svg(report)
    desc = re.search(r"<desc>([^<]*)</desc>", svg).group(1)
    assert "COULD NOT CHECK" in desc
    assert not re.search(r"\d+ of \S+ metric\(s\) show a significant disparity", desc), desc


def test_caller_supplied_interval_is_still_honoured():
    """``confidence_intervals`` with lower/upper is a legitimate way for a CALLER
    to pass a REAL interval in; only the synthesis fallback was removed."""
    svg = adapters_fairness.confidence_intervals_to_svg(
        {
            "metrics": {"calibration_difference": 0.90},
            "confidence_intervals": {"calibration_difference": {"lower": 0.85, "upper": 0.95}},
        }
    )
    assert "[0.850, 0.950]" in svg
    assert "1 significant" in svg


# ── 4. the third state never crashes, and never becomes a fail ──────────────


def test_unmeasurable_metrics_do_not_crash_the_forest_plot():
    """NaN reaches int() in the pixel mapping. It used to raise ValueError and
    take the whole chart down, which is not a third state either."""
    svg = adapters_fairness.confidence_intervals_to_svg(_not_assessable_report())
    assert svg.lstrip().startswith("<svg") or "<svg" in svg


def test_heatmap_cells_do_not_tint_green_when_nothing_was_compared():
    """The heatmap gradient is a claim about SPREAD between groups. With one
    surviving group there is no spread, and ``col_max == col_min`` sent every
    cell down the green branch, so the grid read as a low-disparity all-clear
    underneath its own NOT ASSESSABLE banner."""
    svg = adapters_fairness.disparity_heatmap_to_svg(_not_assessable_report())
    assert "#d1fae5" not in svg, "cells kept the low-disparity green with nothing to compare"
    assert "#065f46" not in svg
    assert "NOT ASSESSABLE" in svg
    # The measured per-group rates are still on the grid; only the colour claim
    # is withdrawn.
    assert "0.53" in svg or "0.52" in svg


def test_could_not_check_is_not_painted_as_a_failure():
    """The third state must not borrow the red of a measured failure either."""
    svg = adapters.fairness_report_to_svg(_not_assessable_report())
    # The FAIL row stripe / badge colour used by the template for a real breach.
    assert "#dc2626" not in svg, "a could-not-check row was painted as a measured failure"
    assert "NO DATA" in svg
