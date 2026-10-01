"""The one way this library says "I could not check that".

WHY THIS MODULE EXISTS
----------------------
The defect it closes has one shape and many instances: a test that could not
run, replaced by the value that reads as its clean answer, and then reported as
if it had run. ``False`` for a detector, ``True`` for a stability check,
``0.0`` for a correlation, ``1.0`` for a p-value, ``"stable"`` for a trend.
Every one of those is a VERDICT on its own scale, and none of them was measured.

It was fixed once properly, in
:meth:`vfairness.multi_agent.negotiation.NegotiationFairnessTracker.analyze`
on 2026-09-08: warn naming what could not be computed and how much of the
input qualified, return ``None`` for the verdict, ``nan`` for the statistics,
and give the not-assessed state its own name in any string field. That fix was
written INLINE, at one call site, so the four siblings carrying the identical
defect were untouched and had to be found again a day later:

    agents/temporal.py                   detect_feedback_loop, detect_drift_cusum,
                                         detect_drift_ewma  (fewer than 3 turns,
                                         and any non-finite turn)
    operations/experimentation/analysis  temporal_stability_check (no period had
                                         enough rows), spillover_detection
    preprocessing/.../significance.py    paired_metric_significance (every
                                         bootstrap delta NaN)
    llm/decodingtrust.py                 _score_privacy_leakage (no prompt text
                                         to compare the response against)

The RESULT SHAPES of those seven sites genuinely differ (a dict, a dataclass,
another dict, a float), so there is no single "not assessed" constructor to
share, and forcing one would be a worse fix than the defect. What they DO share
is the sentence a reader has to see and the name the state is given, and those
are here. A site that imports this cannot warn without stating its counts.

Nothing in here decides anything. It formats one warning and holds one string.
"""

from __future__ import annotations

import warnings

# Fewest rows in a group before that group's calibration error means anything.
#
# WHY IT LIVES HERE rather than beside either surface that uses it. Two surfaces
# draw per-group calibration and they share no code: the matplotlib chart
# (post_processing.calibration.visualization.plot_group_calibration) and the SVG
# (rendering.adapters_calibration.group_calibration_to_svg). They disagreed. The
# chart required ten rows; the SVG required ONE. Measured on 59 rows in group A
# beside a single row in group B, the SVG printed "B  ECE = 0.010", named B the
# BEST CALIBRATED GROUP, reported "Disparity: 0.244" over "2 groups analyzed", and
# recommended applying group-specific calibration, all from one observation, with
# the sample size shown nowhere on the canvas.
#
# The SVG's own docstring already promised the opposite, that a disparity "does not
# exist until two groups have been measured". Its machinery for that was correct and
# complete; the only thing wrong was what it counted as measured, which was
# "entered at least one bin". So the number had to stop being written twice, in two
# packages, where one copy could be raised and the other forgotten.
MIN_ROWS_PER_GROUP_FOR_CALIBRATION = 10

__all__ = ["NOT_ASSESSED", "warn_not_assessed"]


#: The name the could-not-check state carries in any string-valued field.
#:
#: Never ``"stable"``, ``"unknown"``, ``""`` or the absence of the key. Those
#: read as findings, and the point of the state is that there is no finding.
NOT_ASSESSED = "not_assessed"


def warn_not_assessed(
    site: str,
    *,
    measured: int,
    total: int,
    unit: str,
    requirement: str,
    reporting: str,
    instead_of: str,
    stacklevel: int = 3,
) -> None:
    """Warn that a test did not run, and say exactly what is reported instead.

    Every argument is required on purpose. The warning that gets ignored is the
    vague one: "insufficient data" tells a reader neither how much of their
    input was usable nor what the returned object now says. Stating the counts
    and both values makes the caller's next move obvious.

    Args:
        site: Where this happened, as a reader would name it, e.g.
            ``"TemporalTracker.detect_feedback_loop"``.
        measured: How many units of the input could actually be used.
        total: How many units were supplied.
        unit: The counted noun with the clause that qualified it, phrased to
            follow "only M of N", e.g.
            ``"recorded turns had a finite disparity"``.
        requirement: What the test needed and did not get, phrased to follow
            "and", e.g. ``"Kendall's tau needs at least 3"``.
        reporting: The not-assessed values now being returned, e.g.
            ``"has_feedback_loop=None and trend_direction='not_assessed'"``.
        instead_of: The value this used to fabricate, so the change in
            behaviour is legible in the message itself, e.g. ``"False"``.
        stacklevel: Passed through to :func:`warnings.warn`. The default of 3
            points at the caller of the public method, not at this helper and
            not at the method's own body.
    """
    warnings.warn(
        f"{site}: only {measured} of {total} {unit}, and {requirement}. "
        f"Reporting {reporting} (could not check), NOT {instead_of}.",
        UserWarning,
        stacklevel=stacklevel,
    )
