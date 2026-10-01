"""Verdict tests for :mod:`vfairness.rendering.adapters_validation`.

Closes the coverage half of register finding #22 for the data-validation
adapter, which sat at 9.6 percent of 100 statements.

This adapter prints a single word at the top of a data validation report, and
it has three possible values, never two: PASS, FAIL, or UNKNOWN when the input
does not carry enough to assess. Three tests here were recorded as xfail on
2026-08-27 because that word came from a defaulted field rather than from
anything the adapter checked; the defects were fixed on the same day and the
tests below now pin the fixed behaviour.
"""

import warnings

import pytest

from vfairness.rendering import adapters_validation
from vfairness.rendering.adapters_validation import _fmt_val, data_validation_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


def _skinned(original_hex: str) -> str:
    """The colour as it appears in the output, after the Blanco recolour.

    Assertions about colour have to be made against the rendered value, not the
    adapter constant, or they pass while the page shows something else.
    """
    from vfairness.rendering.skins import BLANCO_MAP

    return BLANCO_MAP.get(original_hex.lower(), original_hex.lower())


FAILING_RESULT = {
    "passed": False,
    "summary": "3 issues found",
    "issues": [
        {
            "severity": "CRITICAL",
            "issue_type": "imbalanced_groups",
            "message": "Group 'age>65' has only 12 samples",
            "affected_groups": ["age>65"],
            "recommendation": "Collect more data for the under-represented group",
        },
        {
            "severity": "WARNING",
            "issue_type": "missing_values",
            "message": "Feature 'income' is 8.3 percent missing for 'minority'",
            "affected_groups": ["minority"],
            "recommendation": "Investigate the missing-data pattern",
        },
        {
            "severity": "INFO",
            "issue_type": "data_quality",
            "message": "Dataset contains 5230 records",
            "affected_groups": [],
            "recommendation": "",
        },
    ],
    "metrics": {"total_records": 5230, "missing_rate": 0.032},
}


class TestTheHeadlineVerdict:
    def test_a_failed_validation_prints_fail_and_counts_the_severities(self):
        svg = data_validation_to_svg(FAILING_RESULT)

        assert ">FAIL<" in svg
        assert ">PASS<" not in svg
        assert "1 critical · 1 warn · 1 info" in svg
        assert "3 issues found" in svg

    def test_a_clean_validation_prints_pass_and_says_so_in_the_recommendation(self):
        svg = data_validation_to_svg({"passed": True, "issues": [], "metrics": {}})

        assert ">PASS<" in svg
        assert ">FAIL<" not in svg
        assert "0 critical · 0 warn · 0 info" in svg
        assert "All validation checks passed" in svg

    def test_an_error_severity_is_counted_as_critical_not_as_a_warning(self):
        svg = data_validation_to_svg(
            {
                "passed": False,
                "issues": [{"severity": "ERROR", "issue_type": "t", "message": "m"}],
            }
        )

        assert "1 critical · 0 warn · 0 info" in svg

    def test_an_unrecognised_severity_still_reaches_the_page(self):
        svg = data_validation_to_svg(
            {
                "passed": False,
                "issues": [{"severity": "CATASTROPHIC", "issue_type": "t", "message": "keep me"}],
            }
        )

        assert "keep me" in svg
        assert ">FAIL<" in svg

    def test_an_unrecognised_severity_is_counted_somewhere(self):
        # Was an open defect: the colour lookup fell back to INFO while the three
        # counters matched the exact strings CRITICAL/ERROR/WARNING/INFO, so an
        # unknown severity landed in no bucket at all and the header read
        # "0 critical, 0 warn, 0 info" directly above a listed issue.
        svg = data_validation_to_svg(
            {
                "passed": False,
                "issues": [{"severity": "CATASTROPHIC", "issue_type": "t", "message": "keep me"}],
            }
        )

        assert "0 critical · 0 warn · 0 info" not in svg
        assert "0 critical · 1 warn · 0 info" in svg
        assert "(1 of unrecognised severity)" in svg
        assert "could not be" in svg  # the recommendation says it was not graded

    def test_an_unrecognised_severity_is_not_coloured_as_benign_information(self):
        blue_info = adapters_validation._INFO
        slate_unknown = adapters_validation._UNKNOWN
        rendered_info = _skinned(blue_info)
        rendered_unknown = _skinned(slate_unknown)
        assert rendered_info != rendered_unknown

        svg = data_validation_to_svg(
            {
                "passed": False,
                "issues": [{"severity": "CATASTROPHIC", "issue_type": "t", "message": "keep me"}],
            }
        )

        assert rendered_unknown in svg
        assert rendered_info not in svg

    def test_a_critical_issue_can_never_appear_under_a_pass_headline(self):
        # Was an open defect: `passed` defaulted to True, so a result carrying a
        # CRITICAL issue and no explicit `passed` key rendered a green PASS above
        # the critical issue it had just listed.
        svg = data_validation_to_svg(
            {"issues": [{"severity": "CRITICAL", "issue_type": "t", "message": "m"}]}
        )

        assert ">PASS<" not in svg
        # The library's own rule (validator.py: any error/critical means not
        # passed) is what makes FAIL an assessed verdict here rather than a guess.
        assert ">FAIL<" in svg

    def test_a_missing_pass_flag_without_issues_is_could_not_check_not_a_pass(self):
        svg = data_validation_to_svg({"issues": [], "metrics": {"total_records": 10}})

        assert ">PASS<" not in svg
        assert ">FAIL<" not in svg
        assert ">UNKNOWN<" in svg
        assert "carries no pass/fail flag" in svg

    @pytest.mark.parametrize("flag", [float("nan"), "False", "no", object()])
    def test_a_pass_flag_that_is_not_a_boolean_is_never_truthiness_tested(self, flag):
        # bool("False") and bool(nan) are both True, which is exactly how an
        # unchecked input turns into a green PASS.
        svg = data_validation_to_svg({"passed": flag, "issues": []})

        assert ">PASS<" not in svg
        assert ">UNKNOWN<" in svg
        assert "is not a boolean" in svg

    def test_a_reported_pass_contradicted_by_a_critical_issue_is_refused(self):
        svg = data_validation_to_svg(
            {
                "passed": True,
                "issues": [{"severity": "CRITICAL", "issue_type": "t", "message": "m"}],
            }
        )

        assert ">PASS<" not in svg
        assert ">UNKNOWN<" in svg
        assert "a pass is reported above 1 listed critical/error issue(s)" in svg

    def test_an_unreadable_result_object_is_not_rendered_as_a_pass(self):
        # Was an open defect: an object with no to_dict and no dataclass fields
        # became an empty dict, which then took the `passed` default of True.
        class Unreadable:
            """No to_dict, no dataclass fields: nothing the adapter can use."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = data_validation_to_svg(Unreadable())

        assert ">PASS<" not in svg
        assert ">FAIL<" not in svg
        assert ">UNKNOWN<" in svg
        assert "the result object (Unreadable) could not be read" in svg
        assert any("exposes no to_dict() and no dataclass fields" in str(w.message) for w in caught)

    def test_a_result_whose_to_dict_raises_is_could_not_check_and_warns(self):
        class Exploding:
            def to_dict(self):
                raise RuntimeError("cannot serialise")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = data_validation_to_svg(Exploding())

        assert ">PASS<" not in svg
        assert ">UNKNOWN<" in svg
        assert any("cannot serialise" in str(w.message) for w in caught)


class TestIssueAndMetricRendering:
    def test_issue_text_reaches_the_page_with_its_affected_groups(self):
        svg = data_validation_to_svg(FAILING_RESULT)

        # The template escapes the payload, which is itself the guard against a
        # crafted message breaking out of the text node.
        assert "Group &#39;age&gt;65&#39; has only 12 samples" in svg
        assert "Group 'age>65' has only 12 samples" not in svg
        assert "Collect more data for the under-represented group" in svg

    def test_a_dataclass_issue_is_read_field_by_field(self):
        from dataclasses import dataclass

        @dataclass
        class Issue:
            severity: str
            issue_type: str
            message: str
            affected_groups: list
            recommendation: str

        svg = data_validation_to_svg(
            {
                "passed": False,
                "issues": [Issue("WARNING", "skew", "label skew detected", ["b"], "resample")],
            }
        )

        assert "label skew detected" in svg
        assert "0 critical · 1 warn · 0 info" in svg

    def test_nested_metric_dictionaries_are_flattened_into_cards(self):
        svg = data_validation_to_svg(
            {
                "passed": True,
                "issues": [],
                "metrics": {"group_sizes": {"male": 1200, "female": 980}},
            }
        )

        assert "GROUP SIZES.MALE" in svg
        assert "GROUP SIZES.FEMALE" in svg
        assert "1,200" in svg

    def test_only_the_first_eight_issues_reach_the_page_but_all_are_counted(self):
        issues = [
            {"severity": "WARNING", "issue_type": f"t{i}", "message": f"msg{i}"} for i in range(12)
        ]
        svg = data_validation_to_svg({"passed": False, "issues": issues})

        assert "msg0" in svg
        assert "msg7" in svg
        assert "msg8" not in svg
        assert "0 critical · 12 warn · 0 info" in svg

    def test_an_over_long_summary_is_truncated_instead_of_running_under_the_banner(self):
        svg = data_validation_to_svg({"passed": True, "issues": [], "summary": "x" * 300})

        assert "x" * 97 + "..." in svg
        assert "x" * 101 not in svg

    def test_a_normal_summary_is_passed_through_untouched(self):
        svg = data_validation_to_svg(
            {"passed": True, "issues": [], "summary": "Validation passed with 0 warning(s)"}
        )

        assert ">Validation passed with 0 warning(s)<" in svg

    def test_a_real_result_object_is_read_and_its_verdict_is_kept(self):
        from vfairness.operations.cicd.validator import (
            DataValidationResult,
            ValidationIssue,
            ValidationSeverity,
        )

        result = DataValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    issue_type="imbalanced_groups",
                    severity=ValidationSeverity.CRITICAL,
                    message="group too small",
                )
            ],
            metrics={"total_records": 5230},
            summary="1 issue found",
        )

        svg = data_validation_to_svg(result)

        assert ">FAIL<" in svg
        assert "1 critical · 0 warn · 0 info" in svg
        assert "group too small" in svg
        # ValidationIssue.recommendation is Optional and defaults to None; the
        # word "None" printed as recommendation number 1 in every such report.
        assert "1. None" not in svg
        assert "Resolve the issues listed above" in svg

    def test_the_built_in_demo_renders_a_failing_report(self):
        # example=True is now required: data_validation_to_svg(None) used to
        # reach this fixture as a runtime fallback and rendered its invented
        # FAIL, its five issues and its 5,230 records as a real finding.
        svg = data_validation_to_svg(example=True)

        assert ">FAIL<" in svg
        assert "5 issues found" in svg


class TestValueFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.12345, "0.123"),
            (True, "Yes"),
            (False, "No"),
            (5230, "5,230"),
            ({"a": 1, "b": 2}, "a: 1, b: 2"),
            ({"a": 1, "b": 2, "c": 3, "d": 4}, "4 entries"),
            ([1, 2, 3], "1, 2, 3"),
            ([1, 2, 3, 4], "4 items"),
        ],
    )
    def test_values_are_formatted_for_a_reader(self, value, expected):
        assert _fmt_val(value) == expected

    def test_a_long_string_is_truncated_rather_than_leaking_a_raw_repr(self):
        formatted = _fmt_val("x" * 200)

        assert formatted.endswith("...")
        assert len(formatted) == 60

    def test_bool_is_formatted_before_int_because_bool_is_an_int(self):
        # Regression guard: isinstance(True, int) is True, so a wrong ordering
        # in _fmt_val would print "1" instead of "Yes".
        assert _fmt_val(True) == "Yes"
        assert _fmt_val(1) == "1"


class TestRenderFailureIsReported:
    def test_a_template_failure_is_refused_not_answered_with_an_empty_string(self, monkeypatch):
        """REWRITTEN 2026-09-10, and it reverses a decision taken deliberately.

        The sibling below records that the 2026-08-28 fix closed the missing
        JINJA2 path and left this one alone: "the template-explodes case
        immediately above still returns ''; that is a different path and a
        separate decision." That decision is now reversed, because the reasoning
        the same fix wrote down applies here word for word:

            An empty string is not markup, and "could not render" must not be
            indistinguishable from "rendered nothing worth showing".

        A caller writing the result out gets a ZERO-BYTE file and reads it as
        this run's report, whichever exception produced it. And the warning does
        not rescue it: Python shows a warning once per process, so a batch job
        warns once and then emits nothing, silently, for every report after it.
        Measured 2026-09-10 through the wrapper ten adapters share: three failed
        renders produced ONE warning and three empty strings.

        Raising is also strictly more capable than warning. A batch job that
        wants to skip a broken report can catch the exception; under the old
        behaviour it could not reliably detect one. And the cause is usually a
        jinja2 UndefinedError naming the missing key, which is far more use than
        a warning nobody sees.
        """

        def _boom(template, data):
            raise RuntimeError("template exploded")

        monkeypatch.setattr(adapters_validation, "render_svg", _boom)

        with pytest.raises(RuntimeError, match="template exploded"):
            data_validation_to_svg(FAILING_RESULT)

    def test_a_missing_jinja2_is_refused_not_answered_with_an_empty_string(self, monkeypatch):
        """REWRITTEN 2026-08-28. This test PINNED THE DEFECT.

        It asserted ``svg == ""`` plus a warning, which is the behaviour the
        audit flagged: an empty string is not the "SVG markup string" the
        docstring promises, and a caller writing it out produces a zero-byte
        report. 28 sibling adapters already raised a named ImportError; all 44
        do now. (The template-explodes case immediately above returned "" until
        2026-09-10, when that separate decision was reversed for the same
        reason; see its docstring.)
        """
        monkeypatch.setattr(adapters_validation, "JINJA2_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"vfairness\[rendering\]"):
            data_validation_to_svg(FAILING_RESULT)
