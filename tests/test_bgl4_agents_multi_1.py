"""BGL-4 AUDIT of batch A-agents_multi-1: the OVERTURNED grades, now CLOSED.

Written by the auditor, not by the fixer. Every test here asserts the behaviour
the unit's own docstring promises, so each one was RED against the code as it
stood on 2026-09-27 and carried ``xfail(strict=True)`` to keep the shared suite
green.

CONVERTED 2026-09-27 by the BGL-5 fix wave. The defects below are fixed in
``src/vfairness/agents/temporal.py`` and ``src/vfairness/multi_agent/
groupthink.py``, so the xfail markers are gone and these tests now run as
ordinary pins of the corrected behaviour. The subject and the docstring of each
one are the auditor's; only the marker was removed (and, for the coalition
threshold, the mechanism, because the fix refuses with ValueError instead of a
warning). The fix's own pins, with their sabotage results and over-correction
controls, are in tests/test_bgl5_agents_multi.py.

Grade overturned 1: vfairness.agents.temporal.TemporalTracker.detect_drift_ewma
    was graded PROVEN. A trajectory whose disparity is IDENTICAL at every turn
    has zero standard deviation, so the EWMA control limits have zero width
    (upper_limit == lower_limit == center_line). The recursion
    ``lam * v + (1 - lam) * ewma[-1]`` then lands one ulp off the centre for a
    great many ordinary constant values, and ``ewma_val > upper`` is True at
    every turn: the unit reports has_drift=True with every turn as a drift
    point, on a series that provably never moved, with no warning. Its sibling
    detect_drift_cusum guards exactly this case with an explicit
    ``sigma <= 1e-12`` branch and a comment; detect_drift_ewma has no such
    branch. The named pin never executes the chart path at all (7 of 23 body
    statements under coverage) and still passes when the whole method is made
    to refuse every input, so it cannot tell a working detector from a dead one.

Grade overturned 2: vfairness.multi_agent.groupthink.GroupthinkDetector.analyze_convergence
    was graded PROVEN. GroupthinkResult documents has_groupthink as "None means
    convergence could not be measured OR the trend test could never have fired
    on a series this short, NOT that none was found". At exactly 2 rounds, the
    documented minimum this method accepts, there is no trend test at all and
    the unit answers a two-state bool from a fixed 0.1 endpoint threshold. Two
    rounds of TOTAL agreement (echo_chamber_score 1.0, the strongest echo
    chamber the scale has) and two rounds of TOTAL diversity
    (echo_chamber_score 0.0) both return has_groupthink=False, and no warning
    names that field. The 3-round branch refuses the identical shape with None
    and a warning, for the reason its own comment gives: "the strongest echo
    chamber there is reported as no groupthink". The guard was placed in one
    branch of the dispatch and not the other, and the named pin never executes
    the 2-round branch (lines 219, 220, 226, 227, 234 and 236 are missing under
    coverage of that test file).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.agents.temporal import TemporalTracker
from vfairness.multi_agent.groupthink import GroupthinkDetector


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def test_ewma_does_not_report_drift_on_a_trajectory_that_never_moved():
    """A constant disparity series is the one case where 'no drift' IS the
    measurement, and detect_drift_cusum returns exactly that. EWMA RETURNED
    has_drift=True with every turn flagged (2026-09-27, before the BGL-5 fix);
    it now returns the same measured False its sibling does."""
    tracker = TemporalTracker()
    for turn in range(5):
        tracker.record_turn(turn, [0.005], [0.0])

    trajectory = tracker.compute_trajectory()
    assert [r.value for r in trajectory] == [0.005] * 5
    assert [r.cumulative_drift for r in trajectory] == [0.0] * 5

    cusum, _ = _caught(tracker.detect_drift_cusum)
    assert cusum["has_drift"] is False

    ewma, ewma_messages = _caught(tracker.detect_drift_ewma)
    # The limits have zero width, so the comparison the verdict rests on was
    # never measured. Either verdict a reader could act on would do; inventing
    # drift at every turn in silence is the one that must not happen.
    assert ewma["upper_limit"] == ewma["lower_limit"] == ewma["center_line"]
    assert ewma["has_drift"] is not True, (
        f"has_drift={ewma['has_drift']!r} drift_points={ewma['drift_points']!r} "
        f"on a series that never moved, warnings={ewma_messages!r}"
    )


def test_the_pulse_payload_does_not_publish_drift_for_a_constant_disparity():
    """operations/pulse/agent_probe._temporal_section sets driftDetected=True
    when ANY of the three detectors says True, so the EWMA false positive WAS
    published as a temporal drift finding with detectorsFailed empty
    (2026-09-27, before the BGL-5 fix). This is the consumer half of the pin:
    it fails if the fix is ever reverted inside the detector."""
    from vfairness.operations.pulse.agent_probe import _temporal_section

    rows = []
    stamp = 0
    for _window in range(5):
        for i in range(10):
            rows.append(
                {
                    "timestamp": stamp,
                    "group": "A",
                    "action": "approve" if i < 9 else "escalate",
                }
            )
            stamp += 1
        for _i in range(10):
            rows.append({"timestamp": stamp, "group": "B", "action": "escalate"})
            stamp += 1
    frame = pd.DataFrame(rows)

    section, _ = _caught(
        lambda: _temporal_section(frame, frame["group"], ["A", "B"], "action", [])[0]
    )
    assert [w["disparity"] for w in section["perWindow"]] == [0.9] * 5
    assert section["driftDetected"] is not True, (
        f"driftDetected={section['driftDetected']!r} ewma={section['ewma']!r} "
        f"cusum={section['cusum']!r} on a disparity series that is 0.9 in every window"
    )


def test_two_rounds_of_total_agreement_is_not_a_measured_absence_of_groupthink():
    """Same data at 3 rounds returns None with a warning. At 2 rounds it
    RETURNED False, and the same False as total diversity (2026-09-27, before
    the BGL-5 fix); it now returns None with a warning of its own."""
    detector = GroupthinkDetector()
    identical = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    orthogonal = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    three, three_messages = _caught(
        lambda: detector.analyze_convergence([identical, identical, identical])
    )
    assert three.has_groupthink is None
    assert any("NOT False" in m for m in three_messages)

    two, two_messages = _caught(lambda: detector.analyze_convergence([identical, identical]))
    diverse, _ = _caught(lambda: detector.analyze_convergence([orthogonal, orthogonal]))
    assert two.echo_chamber_score == pytest.approx(1.0)
    assert diverse.echo_chamber_score == pytest.approx(0.0)
    assert two.has_groupthink is None, (
        f"has_groupthink={two.has_groupthink!r} at echo_chamber_score 1.0, the same "
        f"value as {diverse.has_groupthink!r} at echo_chamber_score 0.0; "
        f"warnings={two_messages!r}"
    )


def test_detect_coalitions_refuses_a_threshold_it_cannot_compare_against():
    """Lower severity than the two above, and caller-triggered, but the same
    class. The agreement scale is documented as [0, 1]. A threshold of 1.5 is
    out of reach for ANY data, and ``x >= nan`` is False for every measured
    agreement, so both turn a fully measured agreement matrix into one
    singleton per agent, byte for byte what a measured absence of coalitions
    returns, with no warning. The matrix is guarded by ``np.isfinite``; the
    threshold is guarded by nothing, while four_fifths_rule in this same batch
    does guard a caller-supplied non-finite input and says why.

    MECHANISM CONVERTED 2026-09-27: the auditor asked only that the call stop
    being silent. The BGL-5 fix refuses it outright, the way the sibling
    detectors reject an out-of-range alpha, so the assertion is now
    ``pytest.raises(ValueError)`` rather than "a warning was emitted". The
    subject is unchanged: the same fully measured matrix must not come back as
    one singleton per agent from a comparison no pair could have cleared."""
    detector = GroupthinkDetector()
    matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
    measured, _ = _caught(lambda: detector.detect_coalitions(matrix))
    assert measured == [{0, 1}]

    for threshold in (1.5, float("nan")):
        with pytest.raises(ValueError, match=r"finite agreement score in \[0, 1\]"):
            detector.detect_coalitions(matrix, threshold=threshold)


def test_detect_drift_refuses_a_threshold_it_cannot_compare_against():
    """Grade 16 was SEMI-PROVEN, so this is the sweep of an input class its
    pins do not cover. The BGL-S2 comment inside detect_drift says it exactly:
    "`nan > threshold` is False for Python floats, so an unmeasurable ENDPOINT
    returned the VERDICT False". The fix guarded the DATA side of that
    comparison and not the THRESHOLD side, so a non-finite threshold still
    returns False, the verdict "no drift exceeds the threshold", on a
    trajectory whose measured drift is 0.5, with no warning at all."""
    tracker = TemporalTracker()
    for turn in range(6):
        tracker.record_turn(turn, [0.1 * turn], [0.0])
    measured, measured_messages = _caught(lambda: tracker.detect_drift(0.1))
    assert measured is True and measured_messages == []

    verdict, messages = _caught(lambda: tracker.detect_drift(float("nan")))
    assert verdict is None, (
        f"detect_drift(nan) returned {verdict!r} with warnings={messages!r} on a ramp "
        f"whose first and last measurable disparities are 0.0 and 0.5"
    )
