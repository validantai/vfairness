"""`to_junit_xml()` must emit XML a CI consumer can actually parse.

VF-3. Every field the exporter interpolates is caller-controlled text. A metric
name, a test name and an assertion message routinely carry `&`, `<`, `>` or a
quote, and the exporter built the document with raw f-strings, so those
characters went straight into the markup. Measured 2026-09-07, before the fix,
on a message reading ``gap 0.3 > 0.1 for group "A" & <B>``:

    <failure message="gap 0.3 > 0.1 for group "A" & <B>">
    ElementTree: not well-formed (invalid token): line 3, column 30

The quote is the sharper half of this. `&` and `<` merely make the document
invalid, which is loud. A quote ENDS THE ATTRIBUTE EARLY, so the rest of the
message is parsed as further attributes: caller text becomes markup. That is why
the round-trip assertions below matter more than the parse assertion. A document
can parse and still have lost or moved the content.

Parsing is also the point of the format: a JUnit file that no consumer can read
turns a red fairness gate into a broken build step, which is the failure mode
where a real breach gets waved through as "the reporting is flaky".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from vfairness.operations.cicd.testing import (
    FairnessTestResult,
    FairnessTestSuite,
)
from vfairness.operations.cicd.testing import (
    TestStatus as Status,  # aliased: pytest tries to COLLECT a module-level name starting "Test"
)

# Every XML metacharacter, plus both quote styles, in one payload.
NASTY = "gap 0.3 > 0.1 for \"A\" & <B> & 'C'"
NASTY_NAME = 'parity for "A" & <B>'
NASTY_METRIC = "demographic_parity & ratio <v2>"


def _suite(status: Status) -> FairnessTestSuite:
    suite = FairnessTestSuite(protected_attributes=["g"])
    suite.results = [
        FairnessTestResult(
            test_name=NASTY_NAME,
            metric_name=NASTY_METRIC,
            status=status,
            actual_value=0.3,
            threshold=0.1,
            message=NASTY,
        )
    ]
    return suite


@pytest.mark.parametrize("status", [Status.FAILED, Status.ERROR, Status.SKIPPED, Status.PASSED])
def test_output_parses_for_every_status(status: Status):
    """Each status takes a different branch, and each one interpolates."""
    ET.fromstring(_suite(status).to_junit_xml())


@pytest.mark.parametrize("status", [Status.FAILED, Status.ERROR, Status.SKIPPED])
def test_the_message_survives_the_round_trip_unchanged(status: Status):
    """Lossless, not merely well-formed.

    Escaping that dropped or mangled the payload would still parse, and the
    reader would silently get a different message than the gate produced.
    """
    root = ET.fromstring(_suite(status).to_junit_xml())
    tag = {
        Status.FAILED: "failure",
        Status.ERROR: "error",
        Status.SKIPPED: "skipped",
    }[status]
    node = root.find(f".//{tag}")
    assert node is not None, f"no <{tag}> element was emitted"
    assert node.get("message") == NASTY


def test_the_test_name_survives_the_round_trip_unchanged():
    root = ET.fromstring(_suite(Status.FAILED).to_junit_xml())
    case = root.find(".//testcase")
    assert case is not None
    assert case.get("name") == NASTY_NAME
    assert case.get("classname") == "FairnessTestSuite"


def test_the_metric_name_survives_in_the_failure_body():
    root = ET.fromstring(_suite(Status.FAILED).to_junit_xml())
    body = root.find(".//failure").text or ""
    assert f"Metric: {NASTY_METRIC}" in body


def test_a_quote_cannot_escape_the_attribute_and_become_markup():
    """The injection case, which is the one a parse check alone would miss."""
    suite = FairnessTestSuite(protected_attributes=["g"])
    suite.results = [
        FairnessTestResult(
            test_name="t",
            metric_name="m",
            status=Status.ERROR,
            message='x" injected="yes',
        )
    ]
    node = ET.fromstring(suite.to_junit_xml()).find(".//error")
    assert node.get("injected") is None, "caller text was parsed as an attribute"
    assert node.get("message") == 'x" injected="yes'


def test_none_and_numeric_fields_do_not_crash_the_exporter():
    suite = FairnessTestSuite(protected_attributes=["g"])
    suite.results = [
        FairnessTestResult(
            test_name="t",
            metric_name="m",
            status=Status.FAILED,
            actual_value=None,
            threshold=None,
            message="",
        )
    ]
    ET.fromstring(suite.to_junit_xml())


def test_the_payload_really_contains_metacharacters():
    """NON-VACUITY. If NASTY were ever softened to plain text, every assertion
    above would pass while testing nothing."""
    for ch in ("&", "<", ">", '"', "'"):
        assert ch in NASTY, f"the payload no longer contains {ch!r}"
