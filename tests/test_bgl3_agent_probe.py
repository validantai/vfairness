"""The agent probe may not convert a detector's refusal into a clean verdict.

WHY THIS FILE EXISTS, and why it is the most structural finding of the BGL3 campaign.
The campaign spent its effort making detectors honest: agents/temporal.py now returns
``has_feedback_loop=None``, ``trend_direction='not_assessed'`` and NaN statistics when it
cannot measure, rather than False and 0.0. This module is the layer that publishes those
detectors into a Pulse payload, and it undid all of it:

  * The defaults handed to ``_quiet`` were the CLEAN answers. A detector that RAISED was
    published as ``has_feedback_loop=False, trend_direction="stable",
    trend_strength=0.0, p_value=1.0, has_drift=False, max_cusum=0.0``. Every single
    value the reassuring one.
  * ``bool(feedback.get("has_feedback_loop"))`` turned an honest None into False, and
    False on a fairness verdict reads as "checked, nothing found".
  * ``driftDetected`` was ``bool(None or None or None)``, which is False: the answer that
    closes the question.
  * ``float(x or 0.0)`` on a NaN statistic survived only because NaN happens to be
    truthy, which is not a property to rely on.

The module already had ``_measured_verdict``, whose own docstring says "Three states,
never two. bool(None) is False", and this section did not call it.

REACHABILITY, stated honestly rather than dramatised. The probe refuses to call the
detectors below ``_MIN_TEMPORAL_WINDOWS`` windows, and that constant equals the
detectors' own ``_MIN_TURNS_FOR_TREND``, so the detectors' insufficient-data refusal is
exactly unreachable from this path TODAY. What is reachable today is the exception path.
The refusal path is one constant-change away, in either of two files that each write
that number independently, which is the same shape as the calibration threshold that
was 10 on one surface and 1 on another. test_the_probe_window_minimum_still_covers_the
_detector_minimum below is what turns that from an argument into a check.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.agents as agents_pkg
from vfairness.agents.temporal import _MIN_TURNS_FOR_TREND
from vfairness.operations.pulse import agent_probe as AP

_N = 480


@pytest.fixture(scope="module")
def traces():
    """Groups INTERLEAVED, not blocked. With contiguous blocks most ordered windows
    contain a single group, every such window is skipped, and the section returns early
    without ever reaching the detectors, so the test would pass over nothing."""
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "timestamp": np.arange(_N),
            "group": np.where(np.arange(_N) % 2 == 0, "a", "b"),
            "tool": np.where(rng.rand(_N) < 0.5, "approve", "deny"),
        }
    )


class _Raises:
    """A tracker whose detectors fail, which is the path the defaults are published on."""

    def record_turn(self, *a, **k):
        pass

    def detect_feedback_loop(self, **k):
        raise RuntimeError("synthetic detector failure")

    def detect_drift_cusum(self, **k):
        raise RuntimeError("synthetic detector failure")

    def detect_drift_ewma(self, **k):
        raise RuntimeError("synthetic detector failure")


class _Refuses:
    """A tracker whose detectors REFUSE, returning what temporal.py really returns when
    it cannot measure. Distinct from _Raises on purpose: a refusal and a crash both
    leave the verdict null and only one of them says the data was insufficient."""

    def record_turn(self, *a, **k):
        pass

    def detect_feedback_loop(self, **k):
        return {
            "has_feedback_loop": None,
            "trend_direction": "not_assessed",
            "trend_strength": float("nan"),
            "p_value": float("nan"),
        }

    def detect_drift_cusum(self, **k):
        return {"has_drift": None, "drift_point": None, "max_cusum": float("nan")}

    def detect_drift_ewma(self, **k):
        return {"has_drift": None, "drift_points": []}


def _section(traces, tracker_cls=None):
    original = agents_pkg.TemporalTracker
    if tracker_cls is not None:
        agents_pkg.TemporalTracker = tracker_cls
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return AP._temporal_section(traces, traces["group"], ["a", "b"], "tool", [])
    finally:
        agents_pkg.TemporalTracker = original


def _reached_the_detectors(section) -> bool:
    return "feedbackLoop" in section


@pytest.mark.parametrize("tracker_cls", [_Raises, _Refuses], ids=["raised", "refused"])
def test_a_detector_that_reached_no_verdict_is_not_published_as_a_clean_one(traces, tracker_cls):
    section, findings = _section(traces, tracker_cls)
    assert _reached_the_detectors(section), (
        f"the fixture never reached the detectors: {section.get('reason')}"
    )
    assert section["driftDetected"] is None, (
        "no detector reached a verdict, so drift is unknown. False here is the reading "
        "that closes the question."
    )
    assert section["feedbackLoop"]["hasFeedbackLoop"] is None
    assert section["cusum"]["hasDrift"] is None
    assert section["ewma"]["hasDrift"] is None
    assert section["feedbackLoop"]["trendStrength"] is None
    assert section["feedbackLoop"]["pValue"] is None, (
        "a p-value of 1.0 is the LEAST significant reading on the scale and was the "
        "published default"
    )
    assert section["cusum"]["maxCusum"] is None
    assert section["feedbackLoop"]["trendDirection"] == "not_assessed", (
        "'stable' was the published default, and for a drift test stable is the good news"
    )
    assert not findings, "nothing may be concluded from three detectors that said nothing"


def test_a_crash_is_distinguishable_from_a_refusal(traces):
    """Both leave the verdict null. Only one means the data was insufficient, and a
    reader who cannot tell them apart cannot act on either."""
    crashed, _ = _section(traces, _Raises)
    refused, _ = _section(traces, _Refuses)
    assert crashed["detectorsFailed"] == ["feedbackLoop", "cusum", "ewma"]
    assert refused["detectorsFailed"] == [], (
        "a detector that returned its own could-not-check did not fail, and saying it "
        "did would send a reader looking for a bug instead of for more data"
    )


def test_the_internal_failure_marker_never_reaches_the_payload(traces):
    section, _ = _section(traces, _Raises)
    assert AP._DETECTOR_FAILED not in json.dumps(section, default=str)


def test_a_measurable_series_still_gets_a_real_verdict(traces):
    """Over-correction control, and the reason it has to be here: a section that
    published None for everything would satisfy every assertion above."""
    section, _ = _section(traces)
    assert _reached_the_detectors(section), section.get("reason")
    assert section["driftDetected"] in (True, False), (
        "a real series must produce a real verdict, not a refusal"
    )
    # AND it must AGREE with the three verdicts it summarises. Asserting only that it
    # is a bool let a sabotage through: forcing the summary to False while the
    # detectors had found drift still satisfied `in (True, False)`, so the control
    # accepted the summary contradicting its own inputs. The relationship is the
    # invariant worth pinning, not the type.
    parts = [
        section["feedbackLoop"]["hasFeedbackLoop"],
        section["cusum"]["hasDrift"],
        section["ewma"]["hasDrift"],
    ]
    assert section["driftDetected"] is (True if any(p is True for p in parts) else False), (
        f"driftDetected is {section['driftDetected']} while its three inputs are "
        f"{parts}, so the summary contradicts what the detectors said"
    )
    assert section["detectorsFailed"] == []
    assert isinstance(section["feedbackLoop"]["hasFeedbackLoop"], bool)
    assert isinstance(section["feedbackLoop"]["pValue"], float)
    assert isinstance(section["feedbackLoop"]["trendStrength"], float)
    assert section["feedbackLoop"]["trendDirection"] != "not_assessed"


def test_the_probe_window_minimum_still_covers_the_detector_minimum():
    """The reachability argument above, as a check rather than a paragraph.

    The probe returns early below _MIN_TEMPORAL_WINDOWS windows, and each window that
    counts contributes one turn, so the detectors' insufficient-data refusal is
    unreachable from this path only while the probe's minimum is at least the detector's.
    Those two numbers live in two different files and neither references the other. If
    the detector's rises, this fails and says so, instead of the probe silently starting
    to publish a refusal as a verdict.

    The three-state handling above makes that safe rather than wrong, which is the point:
    this check protects the ARGUMENT, not the behaviour.
    """
    assert AP._MIN_TEMPORAL_WINDOWS >= _MIN_TURNS_FOR_TREND, (
        f"the probe calls the detectors with as few as {AP._MIN_TEMPORAL_WINDOWS} turns "
        f"while they need {_MIN_TURNS_FOR_TREND}, so their refusal path is now live from "
        f"this surface. That is handled honestly, but the comment claiming it is "
        f"unreachable is no longer true and should be corrected."
    )
