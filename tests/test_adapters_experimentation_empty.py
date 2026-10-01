"""The empty-input guards for :mod:`vfairness.rendering.adapters_experimentation`.

Two partial fixes from earlier waves are closed here.

``experiment_results_to_svg({})`` already said COULD NOT CHECK for the p-value
and for the heterogeneity test, but the headline still read "OVERALL TREATMENT
EFFECT +0.0000" and the accessible description still read "Overall treatment
effect +0.0000 across 0 intersections". A point estimate invented from an empty
dict and set beside two honest could-not-check panels is worse than either
failure alone: the honest panels are what make the fabricated headline look
verified, and it is the largest number on the canvas.

``power_analysis_to_svg([])`` rendered "POWERED 0 / 0" behind a GREEN accent,
"AVG POWER 0%", "TOTAL REQUIRED N 0" and "Total N = 0", described as "0 of 0
intersections are underpowered (avg power 0%, target 80%)" at severity INFO with
the action "Collect more data for the under-powered intersections". Zero of zero
underpowered is not a clean bill of health and an average of nothing is not 0
percent. The same defaults applied per row: an intersection that reported no
power, no required N and no sample counts rendered 0% in red with a NEEDS DATA
badge, which says the subgroup was measured and found to have no power at all.

Every test below is paired with a positive control, because a guard that cannot
distinguish a measured zero from an absence has replaced one false claim with
another: a supplied power of 0.0, a supplied effect of +0.0000 and a supplied
sample count of 0 are measurements and must still render as numbers.
"""

import re

import pytest

from vfairness.rendering import adapters_experimentation
from vfairness.rendering.adapters_experimentation import (
    COULD_NOT_CHECK_TEXT,
    P_NOT_COMPUTED,
    experiment_results_to_svg,
    power_analysis_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


# ── fixtures ────────────────────────────────────────────────────────────────

MEASURED_EXPERIMENT = {
    "overall_effect": 0.0512,
    "overall_ci": (0.0104, 0.0920),
    "overall_p_value": 0.0137,
    "heterogeneity_detected": True,
    "heterogeneity_p_value": 0.0021,
    "n_intersections": 1,
    "design_type": "independent",
    "metadata": {"protected_attributes": ["gender"]},
    "intersection_effects": [
        {
            "intersection": ("female", "under_30"),
            "effect_size_d": 0.31,
            "ci_lower": 0.09,
            "ci_upper": 0.53,
            "p_value": 0.004,
            "n_control": 220,
            "n_treatment": 240,
            "significant": True,
            "powered": True,
        }
    ],
}

# A measured NULL result: the experiment ran and moved nothing.
MEASURED_ZERO_EXPERIMENT = {
    **MEASURED_EXPERIMENT,
    "overall_effect": 0.0,
    "overall_ci": (-0.02, 0.02),
    "overall_p_value": 0.87,
}

MEASURED_POWER = [
    {
        "intersection": ("female", "under_30"),
        "power": 0.42,
        "required_n": 800,
        "n_control": 100,
        "n_treatment": 100,
        "effect_size": 0.3,
        "is_powered": False,
    },
    {
        "intersection": ("male", "over_50"),
        "power": 0.91,
        "required_n": 300,
        "n_control": 400,
        "n_treatment": 400,
        "effect_size": 0.5,
        "is_powered": True,
    },
]

# Every intersection adequately powered: the only shape entitled to the green
# accent, and the reference the could-not-check colour is compared against.
ALL_POWERED = [
    {
        "intersection": ("male", "over_50"),
        "power": 0.91,
        "required_n": 300,
        "n_control": 400,
        "n_treatment": 400,
        "effect_size": 0.5,
        "is_powered": True,
    }
]

# A MEASURED zero: the analyser reported a power of exactly 0.0 for a real, tiny
# sample. Not an absence.
MEASURED_ZERO_POWER = [
    {
        "intersection": ("nonbinary", "30_50"),
        "power": 0.0,
        "required_n": 5000,
        "n_control": 4,
        "n_treatment": 6,
        "effect_size": 0.0,
        "is_powered": False,
    }
]

# One measured row and one row that reported nothing at all.
PARTIAL_POWER = [
    MEASURED_POWER[0],
    {"intersection": ("male", "over_50")},
]


def _capture_template_data(monkeypatch):
    captured = {}

    def _capture(template, data):
        captured["template"] = template
        captured.update(data)
        return "<svg></svg>"

    monkeypatch.setattr(adapters_experimentation, "render_svg", _capture)
    return captured


def _desc(svg):
    m = re.search(r"<desc>([^<]*)</desc>", svg)
    assert m, "no accessible <desc> was injected"
    return m.group(1)


#: The 2.5px accent stripe down the left edge of a 66px summary card, which is
#: where a reader takes the grade from before reading a word. Matched by
#: geometry rather than by colour so the assertions can compare one render's
#: stripe against another's, whatever the active skin recolours them to. The
#: corner radius is optional because a skin may flatten it away.
_STRIPE_RE = r'<rect x="36" y="{y}" width="2\.5" height="66"(?: rx="1")? fill="([^"]+)"'


def _stripe_fill(svg, y):
    m = re.search(_STRIPE_RE.format(y=y), svg)
    assert m, f"no accent stripe found at y={y}"
    return m.group(1)


#: The first summary card starts immediately under the 140px header on both
#: templates.
SUMMARY_Y = 140

#: The heterogeneity card on an experiment canvas with no per-intersection rows:
#: 140 header + 66 summary + 16 + 50 empty table + 16 + 200 forest + 16.
EMPTY_HET_Y = 504


# ── defect 1: the experiment headline ───────────────────────────────────────


class TestEmptyExperimentInventsNoTreatmentEffect:
    def test_no_point_estimate_is_printed(self):
        svg = experiment_results_to_svg({})

        assert "+0.0000" not in svg, "a treatment effect was fabricated from an empty dict"
        assert "OVERALL TREATMENT EFFECT" in svg
        assert "no overall treatment effect was reported for this experiment" in svg

    def test_the_headline_says_could_not_check_where_a_reader_sees_it(self):
        svg = experiment_results_to_svg({})

        # Not only in the <desc>: the canvas itself has to carry it, and the
        # heading it sits under is still the effect heading.
        head = svg[svg.index("OVERALL TREATMENT EFFECT") : svg.index("INTERSECTION<")]
        assert COULD_NOT_CHECK_TEXT in head

    def test_the_headline_wears_the_could_not_check_colour_not_a_verdict_colour(self):
        """The accent stripe is read before the words.

        An absent estimate must take the same neutral stripe the heterogeneity
        panel already uses for its own could-not-check state, so one canvas
        speaks with one voice. Comparing the two stripes inside a single render
        keeps this independent of the active skin's palette.
        """
        svg = experiment_results_to_svg({})

        assert _stripe_fill(svg, SUMMARY_Y) == _stripe_fill(svg, EMPTY_HET_Y)

    def test_a_measured_experiment_keeps_a_verdict_colour(self):
        """Positive control for the stripe: a real result is still graded."""
        svg = experiment_results_to_svg(MEASURED_EXPERIMENT)
        het_y = 140 + 66 + 16 + (40 + 36 + 10) + 16 + 200 + 16

        assert _stripe_fill(svg, SUMMARY_Y) != _stripe_fill(svg, het_y)

    def test_the_description_makes_no_claim_the_canvas_does_not(self):
        desc = _desc(experiment_results_to_svg({}))

        assert "+0.0000" not in desc
        assert "across 0 intersections" not in desc
        assert desc.startswith("Experiment Results: " + COULD_NOT_CHECK_TEXT)
        assert "This is not a finding of no effect" in desc

    def test_a_measured_effect_still_leads_the_canvas_and_the_description(self):
        """Positive control: the fix must not withhold a real estimate."""
        svg = experiment_results_to_svg(MEASURED_EXPERIMENT)

        assert "+0.0512" in svg
        assert COULD_NOT_CHECK_TEXT not in svg
        assert "Overall treatment effect +0.0512" in _desc(svg)

    def test_a_measured_effect_of_exactly_zero_is_still_printed(self):
        """A null result is a result. Suppressing it would swap one false claim
        for another: the experiment ran and moved nothing."""
        svg = experiment_results_to_svg(MEASURED_ZERO_EXPERIMENT)

        assert "+0.0000" in svg
        assert COULD_NOT_CHECK_TEXT not in svg
        assert "no overall treatment effect was reported" not in svg

    def test_the_effect_state_is_tracked_as_its_own_flag(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)
        experiment_results_to_svg({})
        assert captured["overall_effect_known"] is False
        assert captured["overall_effect"] == P_NOT_COMPUTED
        assert captured["not_assessable"] is True

        measured = _capture_template_data(monkeypatch)
        experiment_results_to_svg(MEASURED_ZERO_EXPERIMENT)
        assert measured["overall_effect_known"] is True
        assert measured["overall_effect"] == "+0.0000"
        assert measured["not_assessable"] is False

    def test_a_canvas_with_one_real_row_is_not_declared_unassessable(self, monkeypatch):
        """Only a canvas with nothing at all on it is a whole could-not-check.
        One measured row keeps its own prose."""
        captured = _capture_template_data(monkeypatch)

        experiment_results_to_svg(
            {"intersection_effects": [{"intersection": ("a",), "effect_size_d": 0.3}]}
        )

        assert captured["not_assessable"] is False


class TestEmptyExperimentCountsNothing:
    """A count of zero is not a finding of zero, on the same canvas."""

    def test_no_power_summary_of_zero_out_of_zero(self):
        svg = experiment_results_to_svg({})

        assert "0 / 0" not in svg
        assert "intersections adequately powered" not in svg
        assert "no intersection was reported, so none was graded for power" in svg

    def test_no_sample_counts_of_zero_and_no_design_nobody_reported(self):
        svg = experiment_results_to_svg({})

        row = svg[svg.index("N CONTROL") :]
        assert ">0<" not in row, "an empty arm was reported as a measured sample size"
        assert "Independent" not in svg, "a design nobody supplied was named as fact"
        assert svg.count("not reported") >= 4

    def test_a_reported_experiment_keeps_every_count(self):
        """Positive control: real counts, and a real design, still print."""
        svg = experiment_results_to_svg(MEASURED_EXPERIMENT)

        assert "1 / 1" in svg
        assert "intersections adequately powered" in svg
        assert ">220<" in svg and ">240<" in svg
        assert "Independent" in svg


# ── defect 2: the power analysis ────────────────────────────────────────────


class TestEmptyPowerAnalysisGradesNothing:
    def test_no_powered_count_of_zero_out_of_zero(self):
        svg = power_analysis_to_svg([])

        assert "0 / 0" not in svg
        assert "intersections ≥ 80% power" not in svg
        assert "no intersection was supplied" in svg

    def test_no_average_of_nothing(self):
        svg = power_analysis_to_svg([])

        assert "across all intersections" not in svg
        assert "no power value was reported" in svg

    def test_no_required_sample_size_of_zero(self):
        svg = power_analysis_to_svg([])

        assert "for 80% power everywhere" not in svg
        assert "no required sample size was derived" in svg
        assert "Total N = 0" not in svg
        assert f"Total N = {P_NOT_COMPUTED}" in svg

    def test_every_card_says_could_not_check(self):
        svg = power_analysis_to_svg([])

        assert svg.count(COULD_NOT_CHECK_TEXT) >= 3

    def test_the_accent_is_not_the_green_of_an_adequately_powered_analysis(self):
        """A could-not-check card may not borrow a verdict colour.

        The green stripe is calibrated from a render that has actually earned
        it, so this holds whatever the active skin recolours it to.
        """
        empty = power_analysis_to_svg([])
        green = power_analysis_to_svg(ALL_POWERED)

        assert _stripe_fill(empty, SUMMARY_Y) != _stripe_fill(green, SUMMARY_Y)

    def test_an_all_powered_analysis_still_earns_its_green(self):
        """Positive control for the same stripe."""
        green = power_analysis_to_svg(ALL_POWERED)
        amber = power_analysis_to_svg(MEASURED_POWER)

        assert _stripe_fill(green, SUMMARY_Y) != _stripe_fill(amber, SUMMARY_Y)

    def test_the_description_does_not_report_a_finding(self):
        desc = _desc(power_analysis_to_svg([]))

        assert "0 of 0" not in desc
        assert "avg power 0%" not in desc
        assert "are underpowered" not in desc
        assert desc.startswith("Statistical Power Analysis: " + COULD_NOT_CHECK_TEXT)
        assert "not a finding that every intersection is adequately powered" in desc
        assert "(severity: INFO)" not in desc

    def test_the_action_no_longer_tells_the_reader_to_collect_more_data(self):
        svg = power_analysis_to_svg([])

        assert "Collect more data for the under-powered intersections" not in svg

    def test_a_measured_analysis_keeps_its_numbers_and_its_action(self):
        """Positive control for the whole canvas."""
        svg = power_analysis_to_svg(MEASURED_POWER)
        desc = _desc(svg)

        assert "1 / 2" in svg
        assert ">42%<" in svg and ">91%<" in svg
        assert "intersections ≥ 80% power" in svg
        assert "across all intersections" in svg
        assert "for 80% power everywhere" in svg
        assert COULD_NOT_CHECK_TEXT not in svg
        assert "1 of 2 intersections are underpowered" in desc
        assert "Collect more data for the under-powered intersections" in svg

    def test_a_measured_power_of_zero_is_still_a_measurement(self):
        """The mirror failure. A supplied 0.0 means this intersection has no
        power to detect anything, which is exactly what a power analysis is run
        to find, and it must keep its number, its bar and its NEEDS DATA badge."""
        svg = power_analysis_to_svg(MEASURED_ZERO_POWER)

        assert ">0%<" in svg
        assert "NEEDS DATA" in svg
        assert COULD_NOT_CHECK_TEXT not in svg
        assert ">0.000<" in svg
        assert "1 of 1 intersections are underpowered" in _desc(svg)


class TestPowerRowsThatReportedNothing:
    """The per-row remainder: power, required_n and the two arm counts."""

    def test_a_row_with_no_power_is_not_called_underpowered(self):
        svg = power_analysis_to_svg(PARTIAL_POWER)

        assert svg.count("NEEDS DATA") == 1, "a row that reported nothing was graded"
        assert COULD_NOT_CHECK_TEXT in svg

    def test_a_row_with_no_power_shows_no_percentage_and_no_bar(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(PARTIAL_POWER)

        blank = captured["intersections"][1]
        assert blank["power_known"] is False
        assert blank["power_display"] == P_NOT_COMPUTED
        assert blank["powered_known"] is False
        assert blank["required_n"] == P_NOT_COMPUTED
        assert blank["current_n"] == P_NOT_COMPUTED

    def test_an_ungraded_row_is_left_out_of_every_total(self, monkeypatch):
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(PARTIAL_POWER)

        # n_total is the ASSESSED population, as in adapters_fairness: the
        # description says "u of n_total are underpowered", and a row nobody
        # graded must not be counted there as adequately powered.
        assert captured["n_supplied"] == 2
        assert captured["n_total"] == 1
        assert captured["n_ungraded"] == 1
        assert captured["n_underpowered"] == 1
        assert captured["avg_power"] == pytest.approx(0.42)
        assert captured["total_required_n"] == "800"
        assert captured["total_current_n"] == "200"

    def test_the_description_does_not_absolve_the_ungraded_row(self):
        desc = _desc(power_analysis_to_svg(PARTIAL_POWER))

        assert "1 of 2 intersections are underpowered" not in desc
        assert "1 of 1 intersections are underpowered" in desc

    def test_a_supplied_verdict_without_a_power_value_is_still_a_verdict(self, monkeypatch):
        """Positive control for the verdict rule: the analyser's own flag wins,
        and the power cell alone reads could-not-check."""
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg([{"intersection": ("a",), "is_powered": True}])

        row = captured["intersections"][0]
        assert row["powered_known"] is True
        assert row["is_powered"] is True
        assert row["power_known"] is False
        assert captured["n_powered"] == 1
        assert captured["n_total"] == 1

    def test_a_derived_verdict_still_comes_from_a_known_power(self, monkeypatch):
        """Positive control: with no flag, a reported power still grades the row
        against the requested target."""
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg([{"intersection": ("a",), "power": 0.85}], power_target=0.90)

        row = captured["intersections"][0]
        assert row["powered_known"] is True
        assert row["is_powered"] is False

    def test_a_reported_zero_sample_count_is_not_treated_as_missing(self, monkeypatch):
        """A row reporting n_control=0 has been measured: nobody landed in the
        control arm. It keeps its 0."""
        captured = _capture_template_data(monkeypatch)

        power_analysis_to_svg(
            [{"intersection": ("a",), "power": 0.1, "required_n": 0, "n_control": 0}]
        )

        row = captured["intersections"][0]
        assert row["current_n"] == "0"
        assert row["current_known"] is True
        assert row["required_n"] == "0"
        assert row["required_known"] is True
        assert captured["total_current_n"] == "0"
        assert captured["total_required_n"] == "0"
