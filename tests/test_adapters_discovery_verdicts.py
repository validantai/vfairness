"""Behavioural tests for :mod:`vfairness.rendering.adapters_discovery`.

Closes the coverage half of register finding #22 for the auto-discovery
adapter, which sat at 14.6 percent of 70 statements.

This page summarises a scan: which attributes look protected, which fairness
checks it broke, and which groups came out ahead. The claim a reader acts on is
the recommendation block, so that is what the tests assert.
"""

import re

import pytest

from vfairness.rendering.adapters_discovery import auto_discovery_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


CANDIDATES = [
    {"name": "zipcode", "confidence": 0.82, "category": "proxy"},
    {"name": "first_language", "confidence": 0.61, "category": "inferred"},
]

HIGH_VIOLATION = {
    "attribute": "gender",
    "metric": "demographic_parity",
    "value": 0.15,
    "threshold": 0.10,
    "severity": "high",
}

EVEN_GROUPS = [
    {"group": "a", "positive_rate": 0.4, "size": 500, "relative_to_overall": 1.05},
    {"group": "b", "positive_rate": 0.39, "size": 480, "relative_to_overall": 0.97},
]

GROUP_ADVANTAGES = [
    {
        "group": "male",
        "positive_rate": 0.45,
        "size": 1200,
        "relative_to_overall": 1.32,
        "severity": "high",
    },
    {
        "group": "female",
        "positive_rate": 0.22,
        "size": 620,
        "relative_to_overall": 0.65,
        "severity": "high",
    },
]


def _recommendations(svg: str) -> str:
    return " ".join(re.findall(r">(\d+\. [^<>]+)<", svg))


class TestScanSummary:
    def test_the_subtitle_counts_what_was_actually_scanned(self):
        svg = auto_discovery_to_svg(CANDIDATES, [HIGH_VIOLATION], GROUP_ADVANTAGES)

        assert "2 attributes · 1 violations · 2 groups" in svg
        assert "1 high severity" in svg

    def test_high_severity_violations_are_counted_separately_from_the_rest(self):
        violations = [
            HIGH_VIOLATION,
            {**HIGH_VIOLATION, "attribute": "race", "severity": "medium"},
            {**HIGH_VIOLATION, "attribute": "age", "severity": "low"},
        ]

        svg = auto_discovery_to_svg(None, violations, None)

        assert "0 attributes · 3 violations · 0 groups" in svg
        assert "1 high severity" in svg
        assert "1 high-severity violation(s)" in _recommendations(svg)

    def test_a_discovered_attribute_is_shown_with_its_confidence_and_category(self):
        svg = auto_discovery_to_svg(CANDIDATES, None, None)

        assert "zipcode" in svg
        assert ">0.82<" in svg
        assert "proxy" in svg
        assert "2 candidate attribute(s) discovered" in svg

    def test_violation_rows_carry_the_value_and_the_threshold_they_broke(self):
        svg = auto_discovery_to_svg(None, [HIGH_VIOLATION], None)

        assert ">0.150<" in svg
        assert ">0.10<" in svg
        assert "HIGH" in svg


class TestGroupAdvantage:
    def test_advantaged_and_disadvantaged_groups_are_both_reported(self):
        svg = auto_discovery_to_svg(None, None, GROUP_ADVANTAGES)

        assert "1 advantaged and 1 disadvantaged group(s) identified." in svg
        assert "ratio=1.32" in svg
        assert "ratio=0.65" in svg

    def test_groups_inside_the_band_are_not_called_advantaged(self):
        even = [
            {"group": "a", "positive_rate": 0.4, "size": 10, "relative_to_overall": 1.05},
            {"group": "b", "positive_rate": 0.4, "size": 10, "relative_to_overall": 0.95},
        ]

        svg = auto_discovery_to_svg(None, None, even)

        assert "advantaged and" not in _recommendations(svg)

    def test_an_advantaged_group_alone_is_not_reported_as_a_disparity_pair(self):
        """The line claims both sides, so it must not fire on one side only."""
        only_advantaged = [
            {"group": "a", "positive_rate": 0.9, "size": 10, "relative_to_overall": 1.9},
            {"group": "b", "positive_rate": 0.5, "size": 10, "relative_to_overall": 1.3},
        ]

        svg = auto_discovery_to_svg(None, None, only_advantaged)

        assert "advantaged and" not in _recommendations(svg)


class TestTruncationDoesNotLoseTheCount:
    def test_at_most_six_candidates_are_drawn_but_all_are_counted(self):
        many = [{"name": f"c{i}", "confidence": 0.5, "category": "x"} for i in range(9)]

        svg = auto_discovery_to_svg(many, None, None)

        assert ">c0<" in svg
        assert ">c5<" in svg
        assert ">c6<" not in svg
        assert "9 attributes ·" in svg
        assert "9 candidate attribute(s) discovered" in svg

    def test_at_most_eight_violations_are_drawn_but_all_are_counted(self):
        many = [{**HIGH_VIOLATION, "attribute": f"a{i}"} for i in range(11)]

        svg = auto_discovery_to_svg(None, many, None)

        assert ">a0<" in svg
        assert ">a7<" in svg
        assert ">a8<" not in svg
        assert "· 11 violations ·" in svg
        assert "11 high severity" in svg


class TestObjectInputs:
    def test_a_group_advantage_object_is_read_through_to_dict(self):
        class GroupAdvantage:
            def to_dict(self):
                return {
                    "group": "night_shift",
                    "positive_rate": 0.11,
                    "size": 300,
                    "relative_to_overall": 0.4,
                }

        svg = auto_discovery_to_svg(None, None, [GroupAdvantage()])

        assert "night_shift" in svg
        assert "ratio=0.4" in svg

    def test_an_unreadable_object_falls_back_to_a_placeholder_row(self):
        class Unreadable:
            pass

        svg = auto_discovery_to_svg([Unreadable()], None, None)

        assert "attr_0" in svg
        assert "unknown" in svg


class TestNothingFound:
    def test_a_genuinely_clean_scan_says_nothing_was_found(self):
        # A clean scan is one that LOOKED. Empty lists for all three parts were
        # accepted as "it looked and found nothing" until 2026-08-27, but with
        # no attribute and no group there was nothing to look AT, so that case
        # is could-not-check and is covered in
        # tests/test_adapters_zero_work_verdicts.py.
        svg = auto_discovery_to_svg([], [], EVEN_GROUPS)

        assert "No significant fairness issues auto-discovered." in svg
        assert "0 attributes · 0 violations · 2 groups" in svg

    def test_a_recorded_violation_is_never_summarised_as_nothing_found(self):
        # Was an open defect: the recommendation block reacted only to HIGH
        # severity, so a medium-severity breach at four times its threshold sat
        # in the table while the line under it read "No significant fairness
        # issues auto-discovered".
        svg = auto_discovery_to_svg(
            None,
            [
                {
                    "attribute": "gender",
                    "metric": "demographic_parity",
                    "value": 0.4,
                    "threshold": 0.1,
                    "severity": "medium",
                }
            ],
            None,
        )

        assert "No significant fairness issues auto-discovered." not in svg
        assert "1 further violation(s) recorded below high severity" in _recommendations(svg)

    def test_a_low_severity_violation_also_suppresses_the_nothing_found_line(self):
        svg = auto_discovery_to_svg([], [{**HIGH_VIOLATION, "severity": "low"}], [])

        assert "No significant fairness issues auto-discovered." not in svg
        assert "1 further violation(s) recorded below high severity" in _recommendations(svg)

    def test_a_scan_that_did_not_run_is_not_reported_as_a_clean_scan(self):
        """None means the scanner did not look; only [] means it looked and found nothing.

        Same class as identify_proxy_features reporting "no proxies" from a scan
        that could not look at all.
        """
        svg = auto_discovery_to_svg(None, [], None)

        assert "No significant fairness issues auto-discovered." not in svg
        assert "Not scanned: attributes, groups." in _recommendations(svg)
        assert "A clean result cannot be reported for what was not scanned." in svg

    def test_the_not_scanned_line_survives_alongside_a_full_set_of_findings(self):
        violations = [HIGH_VIOLATION, {**HIGH_VIOLATION, "severity": "medium"}]

        svg = auto_discovery_to_svg(CANDIDATES, violations, None)

        recs = _recommendations(svg)
        assert "2 candidate attribute(s) discovered" in recs
        assert "1 high-severity violation(s)" in recs
        assert "Not scanned: groups." in recs

    def test_a_scan_that_looked_and_found_nothing_still_says_so(self):
        # The over-correction control for the two tests above: an actual clean
        # scan must keep its clean summary. It has to have looked at something,
        # so it profiles groups; all-empty is the could-not-check case.
        svg = auto_discovery_to_svg([], [], EVEN_GROUPS)

        assert "No significant fairness issues auto-discovered." in svg
        assert "Not scanned" not in svg

    def test_the_demo_path_renders_a_populated_scan(self):
        # example=True is now required: a bare auto_discovery_to_svg() used to
        # reach this fixture as a runtime fallback and rendered it as a finding.
        svg = auto_discovery_to_svg(example=True)

        assert "3 attributes · 4 violations · 6 groups" in svg
        assert "2 high severity" in svg
