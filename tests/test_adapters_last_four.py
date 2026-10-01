"""Guards for the last adapters that fabricated a verdict or raised namelessly.

Wave 10 of the fabrication sweep. Three surfaces:

* ``calibration_report_to_svg`` asserted a FAILURE and an ALL-CLEAR from the
  same absent data on one canvas: an empty ``CalibrationReport`` rendered
  "ECE 0.000 | MCE 0.000 | BRIER 0.000 | CALIBRATION MISCALIBRATED |
  NO DISPARITY | All groups are similarly well-calibrated." across ZERO groups,
  and the accessible ``<desc>`` agreed with both.
* ``bias_audit_to_svg`` rendered an emerald "OVERALL RISK 0% / MINIMAL" for an
  audit whose four modules all returned nothing, and died on
  ``'NoneType' object has no attribute 'historical_findings'`` when handed no
  report at all.
* ``pareto_frontier_to_svg([], [])`` died inside numpy on "zero-size array to
  reduction operation minimum which has no identity".

The rule these pin: a count of zero is not a finding of zero, a sentinel is not
a measurement, and an absence and a measured zero are different claims that must
render differently. Every could-not-check assertion below is paired with a
healthy-input control in ``TestHealthyInputIsUntouched``, so a fix that simply
suppresses the verdict everywhere fails the controls.
"""

from __future__ import annotations

import html
import re

import pytest

pytest.importorskip("jinja2", reason="SVG rendering requires jinja2")

from vfairness.post_processing.calibration.analyzer import CalibrationReport  # noqa: E402
from vfairness.preprocessing.bias_detection.detector import BiasAuditReport  # noqa: E402
from vfairness.rendering.adapters import bias_audit_to_svg, calibration_report_to_svg  # noqa: E402
from vfairness.rendering.adapters_calibration import pareto_frontier_to_svg  # noqa: E402
from vfairness.rendering.skins import apply_skin  # noqa: E402

NAN = float("nan")


# ── helpers ─────────────────────────────────────────────────────────────────


def _skinned(raw: str) -> str:
    """The Blanco hex a raw palette colour becomes in rendered output.

    Asserting on the RAW hex would silently pass forever: render_svg runs
    apply_skin as its last step, so '#dc2626' never appears in any output and
    `assert "#dc2626" not in svg` is vacuously true.
    """
    out = apply_skin(f'<a fill="{raw}"/>', "blanco")
    match = re.search(r'fill="(#[0-9a-fA-F]{6})"', out)
    assert match, f"apply_skin dropped the fill for {raw}"
    return match.group(1).lower()


FAIL_RED = _skinned("#dc2626")
PASS_GREEN = _skinned("#059669")
SLATE = _skinned("#64748b")


def _visible(svg: str) -> str:
    """Every rendered text node, joined. This is what a reader of the SVG sees."""
    body = svg.split("</metadata>")[-1]
    return " | ".join(html.unescape(t) for t in re.findall(r">([^<>]+)<", body) if t.strip())


def _desc(svg: str) -> str:
    """The accessible description a screen reader (or a parser) is given."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "render_svg must always inject an accessible <desc>"
    return html.unescape(match.group(1))


def _fills(svg: str) -> set:
    return {c.lower() for c in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)}


class _Disparity:
    def __init__(self, spread, best, worst):
        self.ece_disparity = spread
        self.least_miscalibrated_group = best
        self.most_miscalibrated_group = worst


def _calibration_report(**over) -> CalibrationReport:
    base = dict(
        timestamp="2026-08-27 10:00",
        data_info={},
        protected_attribute="",
        n_groups=0,
        overall_metrics={},
        group_metrics={},
        disparity_analysis=None,
        brier_decomposition=None,
        tradeoff_analysis=None,
        recommendation=None,
        is_well_calibrated=False,
        has_significant_disparity=False,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return CalibrationReport(**base)


def _healthy_calibration_report() -> CalibrationReport:
    return _calibration_report(
        protected_attribute="gender",
        n_groups=3,
        overall_metrics={"ece": 0.0731, "mce": 0.1902, "brier": 0.1543},
        group_metrics={
            "female": {"ece": 0.0412, "mce": 0.1100, "brier": 0.1401},
            "male": {"ece": 0.0688, "mce": 0.1650, "brier": 0.1502},
            "nonbinary": {"ece": 0.1394, "mce": 0.2810, "brier": 0.1899},
        },
        disparity_analysis=_Disparity(0.0982, "female", "nonbinary"),
        has_significant_disparity=True,
    )


class _Severity:
    def __init__(self, value):
        self.value = value


class _Confidence:
    def __init__(self, confidence_score):
        self.confidence_score = confidence_score


class _Correlated:
    def __init__(self, correlation):
        self.correlation = correlation


class _Representation:
    def __init__(self, attribute, distributions, ratios, severity):
        self.attribute = attribute
        self.group_distributions = distributions
        self.representation_ratios = ratios
        self.severity = _Severity(severity)


def _bias_report(**over) -> BiasAuditReport:
    base = dict(
        timestamp="2026-08-27 10:00",
        dataset_info={},
        protected_attributes=[],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return BiasAuditReport(**base)


def _healthy_bias_report() -> BiasAuditReport:
    return _bias_report(
        dataset_info={"n_rows": 12000, "n_columns": 24},
        protected_attributes=["gender", "age_band"],
        historical_findings=[_Confidence(0.82), _Confidence(0.44)],
        representation_findings=[
            _Representation(
                "gender", {"female": 0.31, "male": 0.69}, {"female": 0.62, "male": 1.38}, "high"
            )
        ],
        disparity_findings=[_Correlated(0.37)],
        proxy_findings=[_Correlated(0.79)],
        overall_risk_score=0.6183,
        critical_issues=[
            {
                "type": "Representation",
                "description": "Women under-represented",
                "details": "31% vs a 50% benchmark",
                "severity": "critical",
            }
        ],
        recommendations=["Rebalance the training set."],
    )


# ── 1. calibration_report_to_svg, the self-contradicting canvas ─────────────


class TestEmptyCalibrationReport:
    def test_it_prints_no_calibration_number_at_all(self):
        # Was: ECE 0.000 | MCE 0.000 | BRIER 0.000, three sentinels read as
        # three measurements.
        seen = _visible(calibration_report_to_svg(_calibration_report()))

        assert "0.000" not in seen
        assert seen.count("N/A") >= 3

    def test_it_asserts_no_calibration_verdict(self):
        # Was: a red MISCALIBRATED badge derived from a defaulted bool with no
        # ECE behind it.
        seen = _visible(calibration_report_to_svg(_calibration_report()))

        assert "MISCALIBRATED" not in seen
        assert "WELL CALIBRATED" not in seen
        assert "NOT ASSESSABLE" in seen

    def test_it_asserts_no_all_clear_either(self):
        # Was: a green "NO DISPARITY / All groups are similarly well-calibrated"
        # printed across ZERO groups, on the same canvas as the failure badge.
        seen = _visible(calibration_report_to_svg(_calibration_report()))

        assert "NO DISPARITY" not in seen
        assert "All groups are similarly well-calibrated" not in seen
        assert "DISPARITY DETECTED" not in seen

    def test_the_reason_is_on_the_canvas_not_only_in_the_desc(self):
        seen = _visible(calibration_report_to_svg(_calibration_report()))

        assert "no overall calibration metric was supplied" in seen
        assert "no group was supplied" in seen

    def test_could_not_check_is_not_rendered_as_a_rejection(self):
        # The third state borrows neither palette: it is not a fail and not a
        # pass. Collapsing it into the red branch is the same fabrication in
        # the other direction.
        svg = calibration_report_to_svg(_calibration_report())

        assert FAIL_RED not in _fills(svg)
        assert PASS_GREEN not in _fills(svg)
        assert SLATE in _fills(svg)

    def test_the_desc_is_never_more_confident_than_the_badge(self):
        # Was: "ECE 0.000 (miscalibrated); ECE disparity 0.000 across groups."
        desc = _desc(calibration_report_to_svg(_calibration_report()))

        assert "COULD NOT CHECK" in desc
        assert "miscalibrated" not in desc
        assert "0.000" not in desc

    def test_a_group_whose_ece_was_never_supplied_is_not_scored_good(self):
        # Was: `metrics.get("ece", 0)` printed 0.000 and lit the green GOOD
        # pill for a group whose calibration error is absent from the dict.
        report = _calibration_report(
            protected_attribute="gender",
            n_groups=2,
            overall_metrics={"ece": 0.031, "mce": 0.09, "brier": 0.14},
            group_metrics={
                "female": {"mce": 0.08},
                "male": {"ece": 0.041, "mce": 0.11, "brier": 0.15},
            },
            disparity_analysis=_Disparity(0.013, "female", "male"),
            is_well_calibrated=True,
        )
        seen = _visible(calibration_report_to_svg(report))

        assert "NOT MEASURED" in seen
        # One GOOD pill, for the group that actually has an ECE.
        assert seen.count("GOOD") == 1
        assert "0.000" not in seen

    def test_a_nan_group_ece_is_not_a_measurement(self):
        report = _calibration_report(
            n_groups=1,
            overall_metrics={"ece": 0.031},
            group_metrics={"female": {"ece": NAN, "mce": NAN, "brier": NAN}},
        )
        seen = _visible(calibration_report_to_svg(report))

        # Word-bounded: "CalibrationAnalyzer" contains the letters n-a-n, and a
        # bare substring test passes forever on that alone.
        assert re.search(r"\bnan\b", seen, re.IGNORECASE) is None
        assert "NOT MEASURED" in seen

    def test_one_measured_group_is_not_a_comparison(self):
        # A spread needs TWO measured groups to exist, and "best"/"worst" are
        # comparative: naming the single supplied group both compares it to
        # nothing.
        report = _calibration_report(
            protected_attribute="gender",
            n_groups=1,
            overall_metrics={"ece": 0.031, "mce": 0.09, "brier": 0.14},
            group_metrics={"female": {"ece": 0.028, "mce": 0.08, "brier": 0.13}},
            disparity_analysis=_Disparity(0.0, "female", "female"),
            is_well_calibrated=True,
        )
        seen = _visible(calibration_report_to_svg(report))

        assert "only one group's calibration error was supplied" in seen
        assert "NO DISPARITY" not in seen
        assert "Best calibrated" not in seen
        assert "Most miscalibrated" not in seen
        # The measured overall ECE is still reported: this is a withheld
        # comparison, not a withheld canvas.
        assert "0.031" in seen
        assert "WELL CALIBRATED" in seen

    def test_a_report_that_is_not_a_report_names_the_missing_input(self):
        with pytest.raises(TypeError) as exc:
            calibration_report_to_svg(None)

        assert "CalibrationReport" in str(exc.value)
        assert "overall_metrics" in str(exc.value)


# ── 2. bias_audit_to_svg, zero of zero and the nameless raise ───────────────


class TestZeroFindingBiasAudit:
    def test_it_withholds_the_overall_risk_percentage(self):
        # Was: an emerald "0%" and a MINIMAL band label, because
        # _calculate_overall_risk returns 0.0 for a run where NO module
        # executed exactly as it does for a dataset every module cleared.
        seen = _visible(bias_audit_to_svg(_bias_report()))

        assert "0%" not in seen
        assert "MINIMAL" not in seen
        assert "NOT ASSESSABLE" in seen

    def test_the_module_tiles_are_not_scored_zero(self):
        # Was: four green "0.00" tiles from _module_score([]) -> 0.0.
        seen = _visible(bias_audit_to_svg(_bias_report()))

        assert "0.00" not in seen
        assert seen.count("N/A") >= 4

    def test_it_is_not_rendered_as_a_rejection_or_a_pass(self):
        svg = bias_audit_to_svg(_bias_report())

        assert PASS_GREEN not in _fills(svg)
        assert FAIL_RED not in _fills(svg)
        assert SLATE in _fills(svg)

    def test_the_reason_is_on_the_canvas_not_only_in_the_desc(self):
        seen = _visible(bias_audit_to_svg(_bias_report()))

        assert "no module of this audit returned a finding" in seen

    def test_the_desc_certifies_nothing(self):
        desc = _desc(bias_audit_to_svg(_bias_report()))

        assert "COULD NOT CHECK" in desc
        assert "MINIMAL" not in desc
        assert "0.00 (" not in desc

    def test_an_empty_issue_table_says_so_rather_than_staying_silent(self):
        seen = _visible(bias_audit_to_svg(_bias_report()))

        assert "No issue was returned" in seen
        assert "No representation finding was returned" in seen

    def test_no_report_at_all_names_the_missing_input(self):
        # Was: AttributeError "'NoneType' object has no attribute
        # 'historical_findings'", which names nothing a caller can act on.
        with pytest.raises(TypeError) as exc:
            bias_audit_to_svg(None)

        message = str(exc.value)
        assert "BiasAuditReport" in message
        assert "NoneType" in message
        assert "historical_findings" in message

    def test_a_dict_is_not_a_report(self):
        with pytest.raises(TypeError) as exc:
            bias_audit_to_svg({"overall_risk_score": 0.5})

        assert "BiasAuditReport" in str(exc.value)


# ── 3. pareto_frontier_to_svg, the numpy internals error ────────────────────


class TestEmptyParetoFrontier:
    def test_no_configuration_renders_a_canvas_instead_of_a_numpy_error(self):
        # Was: ValueError "zero-size array to reduction operation minimum which
        # has no identity", raised from inside numpy's _methods.py.
        svg = pareto_frontier_to_svg([], [])

        assert svg.lstrip().startswith("<svg")
        assert "NOT ASSESSABLE" in _visible(svg)

    def test_it_withholds_the_counts(self):
        # "0 configurations | 0 Pareto-optimal" reads as a search that ran and
        # came back empty. No search ran.
        #
        # Asserted on the SHAPE of the summary line, not on the literal "0":
        # `assert "0 configurations" not in seen` passed even with the summary
        # branch removed, because the withheld counts render as "None
        # configurations | None Pareto-optimal". A guard that cannot fail is
        # not a guard.
        seen = _visible(pareto_frontier_to_svg([], []))

        assert re.search(r"\S+ configurations", seen) is None
        assert "Pareto-optimal" not in seen
        assert "Best trade-off" not in seen
        assert "no configuration was supplied" in seen
        assert "recommends no configuration and rules none out" in seen

    def test_it_is_not_rendered_as_a_rejection(self):
        svg = pareto_frontier_to_svg([], [])

        assert FAIL_RED not in _fills(svg)
        assert SLATE in _fills(svg)

    def test_the_desc_recommends_no_configuration(self):
        desc = _desc(pareto_frontier_to_svg([], []))

        assert "COULD NOT CHECK" in desc
        assert "Pareto-optimal" not in desc

    def test_points_that_cannot_be_placed_do_not_become_a_frontier(self):
        # Every coordinate is NaN, so nothing can be placed on either axis.
        seen = _visible(pareto_frontier_to_svg([NAN, NAN], [0.1, 0.2], labels=["a", "b"]))

        assert "NOT ASSESSABLE" in seen
        assert "finite coordinates" in seen
        # Shape, not the literal count: see test_it_withholds_the_counts.
        assert re.search(r"\S+ configurations", seen) is None

    def test_a_single_unplaceable_point_is_excluded_and_reported(self):
        # The other two points are real, so the chart still renders; the
        # excluded one is named on the canvas rather than silently dropped or
        # crashing the pixel mapping on "cannot convert float NaN to integer".
        seen = _visible(
            pareto_frontier_to_svg([0.10, NAN, 0.05], [0.30, 0.10, 0.40], labels=["m1", "x", "m3"])
        )

        assert "1 of 3 configuration(s) have non-finite coordinates and are excluded" in seen
        assert "2 configurations" in seen
        assert "| x |" not in seen

    def test_the_baseline_marker_follows_its_point_when_others_are_dropped(self):
        # baseline_index refers to the SUPPLIED order. After the unplaceable
        # point at index 0 is dropped, index 1 ("m2") sits at position 0; an
        # un-remapped index silently marks "m3" as the baseline instead.
        #
        # Asserted by reading the diamond's own coordinate and the label
        # written beside it. `assert "polygon" in svg` passed under the
        # sabotage, because the LEGEND draws a diamond too.
        svg = pareto_frontier_to_svg(
            [NAN, 0.20, 0.05], [0.30, 0.10, 0.40], labels=["x", "m2", "m3"], baseline_index=1
        )

        diamond = re.search(r'translate\((\d+),(\d+)\)"?>\s*<polygon points="0,-10', svg)
        assert diamond, "the baseline diamond is not drawn on the chart"
        label = re.search(rf'<text x="{int(diamond.group(1)) + 14}"[^>]*>([^<]+)</text>', svg)
        assert label is not None, "the baseline point carries no label"
        assert label.group(1) == "m2"

    def test_mismatched_coordinate_counts_name_both_lengths(self):
        with pytest.raises(ValueError) as exc:
            pareto_frontier_to_svg([0.1, 0.2], [0.3])

        message = str(exc.value)
        assert "2 calibration_errors" in message
        assert "1 fairness_violations" in message

    def test_a_short_label_list_is_refused_rather_than_mislabelling(self):
        with pytest.raises(ValueError) as exc:
            pareto_frontier_to_svg([0.1, 0.2], [0.3, 0.4], labels=["a"])

        assert "1 labels" in str(exc.value)
        assert "2 point(s)" in str(exc.value)


# ── 4. over-correction control ──────────────────────────────────────────────


class TestHealthyInputIsUntouched:
    """A fix that suppresses the verdict for everyone is not a fix.

    These are the same inputs whose rendered SVG and rasterised PNG were
    diffed against git 17c2db5 and came back byte-identical.
    """

    def test_a_healthy_calibration_report_still_reports_every_number(self):
        seen = _visible(calibration_report_to_svg(_healthy_calibration_report()))

        assert "0.073" in seen and "0.190" in seen and "0.154" in seen
        assert "MISCALIBRATED" in seen
        assert "DISPARITY DETECTED" in seen
        assert "0.098" in seen
        assert "Most miscalibrated: nonbinary · best calibrated: female" in seen
        assert "NOT ASSESSABLE" not in seen
        assert "N/A" not in seen

    def test_a_healthy_calibration_report_keeps_its_per_group_bands(self):
        seen = _visible(calibration_report_to_svg(_healthy_calibration_report()))

        assert "GOOD" in seen and "FAIR" in seen and "POOR" in seen
        assert "NOT MEASURED" not in seen

    def test_a_healthy_calibration_desc_still_states_the_finding(self):
        desc = _desc(calibration_report_to_svg(_healthy_calibration_report()))

        assert "ECE 0.073" in desc
        assert "miscalibrated" in desc
        assert "COULD NOT CHECK" not in desc

    def test_a_healthy_bias_audit_still_scores_and_bands(self):
        seen = _visible(bias_audit_to_svg(_healthy_bias_report()))

        assert "61%" in seen
        assert "MEDIUM" in seen
        assert "NOT ASSESSABLE" not in seen
        assert "N/A" not in seen
        assert "Women under-represented" in seen

    def test_a_healthy_bias_audit_still_scores_every_module(self):
        seen = _visible(bias_audit_to_svg(_healthy_bias_report()))

        for score in ("0.63", "0.75", "0.37", "0.79"):
            assert score in seen, f"module score {score} disappeared from a healthy audit"

    def test_a_healthy_pareto_still_names_the_frontier_and_the_best_tradeoff(self):
        seen = _visible(
            pareto_frontier_to_svg(
                [0.10, 0.20, 0.05, 0.30],
                [0.30, 0.10, 0.40, 0.50],
                labels=["m1", "m2", "m3", "m4"],
            )
        )

        assert "4 configurations  |  3 Pareto-optimal" in seen
        assert "Best trade-off:" in seen and "m2" in seen
        assert "NOT ASSESSABLE" not in seen
        assert "excluded" not in seen

    def test_a_healthy_pareto_keeps_its_threshold_zones(self):
        seen = _visible(
            pareto_frontier_to_svg(
                [0.10, 0.20], [0.05, 0.30], labels=["a", "b"], fairness_threshold=0.15
            )
        )

        assert "PASS ZONE" in seen
        assert "ABOVE THRESHOLD" in seen
