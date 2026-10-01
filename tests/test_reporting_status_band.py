"""The reporting dashboard's headline BAND is a measurement, with three states.

The fourth default, closed 2026-08-28.
``adapters_reporting._from_report`` carries a CRITICAL note forbidding the four
defaults its health-score block used to hold (``score 0``, ``status "yellow"``,
``trend "stable"``, ``slope 0.0``). Three were removed. The fourth was
reinstated ONE LINE BELOW that note by the lookup's own fallback::

    status_color, status_bg, status_label = _STATUS_COLORS.get(status, _STATUS_COLORS["yellow"])

which is ``hs.get("status", "yellow")`` wearing a different hat. Consequences,
all reproduced at HEAD cff5367 before the fix:

* a band that was never reported (``None``), or empty, rendered a GRADED YELLOW
  badge;
* so did any case variant, and ``"RED"`` is the likeliest spelling of all: a
  health score of 31 arriving as ``status="RED"`` rendered **YELLOW** on the
  canvas and severity **MEDIUM** in the accessible ``<desc>``. That is a
  DOWNGRADE of a red report, on both surfaces at once, and the ``<desc>`` is the
  whole artifact for a screen-reader user.

Nothing pinned it: grepping ``_STATUS_COLORS`` across ``tests/`` found only
fixtures passing a valid lowercase ``"yellow"``.

The tests at the bottom are the over-correction control: every band that WAS
reported must render exactly as it did before.
"""

import re

import pytest

from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

_BANDS = ("GREEN", "YELLOW", "RED", "NOT CHECKED")


def _report(status, score=31):
    """A report that measured something real, varying ONLY the band.

    One metric, comfortably inside its threshold, so nothing in this fixture can
    escalate the headline on its own: whatever band comes out is the band the
    status field produced.
    """
    return {
        "tier": "OPERATIONAL",
        "model_name": "m",
        "model_version": "1",
        "metrics": [
            {
                "name": "demographic_parity_difference",
                "value": 0.05,
                "threshold": 0.10,
                "n_affected_groups": 0,
            }
        ],
        "n_metrics": 1,
        "health_score": {
            "score": score,
            "status": status,
            "components": {
                "metric_compliance": 80,
                "alert_frequency": 90,
                "drift_stability": 70,
            },
        },
    }


def _render(status, score=31):
    svg = reporting_dashboard_to_svg(_report(status, score))
    bands = [b for b in _BANDS if f">{b}<" in svg]
    assert len(bands) == 1, f"exactly one headline band expected, got {bands}"
    desc = re.search(r"<desc[^>]*>(.*?)</desc>", svg, re.S)
    assert desc, "the dashboard must always carry an accessible <desc>"
    sev = re.search(r"severity: ([A-Z]+)", desc.group(1))
    assert sev, f"no severity in <desc>: {desc.group(1)!r}"
    return bands[0], sev.group(1), desc.group(1), svg


class TestAMiscasedBandIsNotDowngraded:
    """The damaging direction: a RED report must not render YELLOW."""

    @pytest.mark.parametrize("spelling", ["red", "RED", "Red", "rEd", " red", "red "])
    def test_every_spelling_of_red_renders_red(self, spelling):
        band, severity, desc, _ = _render(spelling)
        assert band == "RED", f"status={spelling!r} rendered {band}, downgrading a red report"
        assert severity == "CRITICAL", f"status={spelling!r} gave <desc> severity {severity}"
        assert "(RED)" in desc

    @pytest.mark.parametrize("spelling", ["yellow", "YELLOW", "Yellow", " yellow "])
    def test_every_spelling_of_yellow_renders_yellow(self, spelling):
        band, severity, _, _ = _render(spelling)
        assert band == "YELLOW"
        assert severity == "MEDIUM"

    @pytest.mark.parametrize("spelling", ["green", "GREEN", "Green"])
    def test_every_spelling_of_green_renders_green(self, spelling):
        band, severity, _, _ = _render(spelling, score=94)
        assert band == "GREEN"
        assert severity == "INFO"


class TestAnUnreportedBandIsNotGraded:
    """The absent direction: no band reported is not a YELLOW band."""

    @pytest.mark.parametrize("missing", [None, "", "   ", "critical", "amber", 3, True])
    def test_a_band_that_was_never_reported_is_not_coloured(self, missing):
        band, _severity, desc, _svg = _render(missing)
        assert band == "NOT CHECKED", (
            f"status={missing!r} rendered the graded band {band}: a band nobody assigned"
        )
        # Not any of the three graded bands, said explicitly, because that is
        # the whole claim and a band list can grow.
        assert band not in ("GREEN", "YELLOW", "RED")

    def test_an_unreported_band_is_stated_in_words_where_a_reader_sees_it(self):
        _band, _sev, _desc, svg = _render(None)
        # The narrative panel is what a reader reads; a slate chip is a colour,
        # not a sentence.
        assert "NOT REPORTED" in svg
        assert "no recognised status band" in svg

    def test_the_missing_key_and_a_present_none_take_the_same_path(self):
        absent = _report(None)
        absent["health_score"].pop("status")
        svg_absent = reporting_dashboard_to_svg(absent)
        _band, _sev, _desc, svg_none = _render(None)
        for token in ("NOT CHECKED", "NOT REPORTED"):
            assert token in svg_absent and token in svg_none

    def test_a_measured_breach_still_escalates_over_an_unreported_band(self):
        """Part (a) of the headline rule: withholding the band must not silence
        a failure this adapter measured itself."""
        d = _report(None)
        d["metrics"] = [
            {
                "name": "demographic_parity_difference",
                "value": 0.45,
                "threshold": 0.10,
                "n_affected_groups": 2,
                "affected_groups": ["female"],
            }
        ]
        svg = reporting_dashboard_to_svg(d)
        assert ">RED<" in svg, "a measured breach must still raise the headline"
        assert "BREACH:" in svg


class TestHealthyInputIsUnchanged:
    """Over-correction control: a correctly reported band renders as before."""

    @pytest.mark.parametrize(
        "status,expected_band,expected_sev,score",
        [
            ("green", "GREEN", "INFO", 94),
            ("yellow", "YELLOW", "MEDIUM", 72),
            ("red", "RED", "CRITICAL", 31),
        ],
    )
    def test_the_three_reported_bands_render_exactly_as_before(
        self, status, expected_band, expected_sev, score
    ):
        band, severity, desc, svg = _render(status, score=score)
        assert band == expected_band
        assert severity == expected_sev
        assert f"health {score}/100 ({expected_band})" in desc
        # And the withheld-band note is NOT written for a report that supplied
        # one: the fix must be silent on healthy input.
        assert "NOT REPORTED" not in svg
        assert "no recognised status band" not in svg
