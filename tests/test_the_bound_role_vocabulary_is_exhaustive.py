"""One rule, one vocabulary, and no role judged by another role's rule.

``vacuous_bound_reason`` decides whether a configured bound can be breached by
any value a metric can take. The answer depends on the ROLE the bound plays, and
the roles point in opposite directions: a THRESHOLD is unusable when
``bound >= worst``, a REQUIRED_IMPROVEMENT when ``bound <= -span``.

WHY THIS FILE EXISTS. Until 2026-09-30 the THRESHOLD case was the FALL-THROUGH,
so every role the function did not handle was judged by the threshold rule.
Measured on values in [0, 1] with bound=60: the string
'required_baseline_floor', the integer 0 and None all came back with the
threshold reason, while the real REQUIRED_IMPROVEMENT role is correctly silent at
that bound. A fourth member added later, or a caller that loses the role, would
have been judged by the wrong rule in the wrong direction, which is the exact
substitution this function exists to prevent.

The enum was also the last public unit the beta gate counted as unexamined, and
this is that examination: every member is consumed, each is flagged in its own
direction and silent in the other, and nothing outside the vocabulary is judged
at all.
"""

from __future__ import annotations

import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    vacuous_bound_reason,
)

#: Values in [0, 1], so span is 1 and the largest magnitude is 1.
METRIC = "demographic_parity_difference"

#: role -> (a bound that IS vacuous for it, a bound that is genuinely usable).
#: The vacuous bound differs per role ON PURPOSE: that difference is the subject.
BY_ROLE = {
    BoundRole.THRESHOLD: (60.0, 0.1),
    BoundRole.REQUIRED_IMPROVEMENT: (-2.0, 0.05),
    BoundRole.ALLOWED_DEGRADATION: (60.0, 0.02),
}


def test_every_member_of_the_vocabulary_is_consumed():
    """If a member is added without a branch, this fails rather than the member
    being silently judged by another member's rule."""
    assert set(BY_ROLE) == set(BoundRole), (
        "BoundRole gained or lost a member and this file was not updated, so the new "
        f"one is unexamined: {set(BoundRole) ^ set(BY_ROLE)}"
    )


@pytest.mark.parametrize("role", list(BoundRole), ids=lambda r: r.name)
def test_each_role_is_flagged_in_its_own_direction(role):
    vacuous, _usable = BY_ROLE[role]
    reason = vacuous_bound_reason(METRIC, vacuous, role)
    assert reason, f"{role.name} did not flag {vacuous}, which it cannot be breached by"
    assert "could not be checked" not in reason.lower(), reason


@pytest.mark.parametrize("role", list(BoundRole), ids=lambda r: r.name)
def test_each_role_is_silent_on_a_usable_bound(role):
    """OVER-CORRECTION CONTROL. A rule that flagged every bound would pass the test
    above and refuse every gate this library runs."""
    _vacuous, usable = BY_ROLE[role]
    assert vacuous_bound_reason(METRIC, usable, role) is None, role.name


def test_the_fail_closed_pole_of_a_required_improvement_stays_silent():
    """A required improvement of 60 on a metric in [0, 1] can never be MET, so the
    check always blocks. That is the safe direction and is deliberately not routed
    to this refusal: doing so would turn a refusal into an approval, which is the
    reasoning already recorded for an infinite group-size minimum."""
    assert vacuous_bound_reason(METRIC, 60.0, BoundRole.REQUIRED_IMPROVEMENT) is None


@pytest.mark.parametrize(
    "outsider",
    ["required_baseline_floor", "threshold", 0, None, True],
    ids=["future_member", "the_value_not_the_member", "zero", "none", "bool"],
)
def test_a_role_outside_the_vocabulary_is_refused_not_judged(outsider):
    """Including the string 'threshold', which equals BoundRole.THRESHOLD's VALUE:
    the rule is keyed on the member, and a caller passing the raw value has not
    passed a role."""
    reason = vacuous_bound_reason(METRIC, 0.1, outsider)
    assert reason, f"{outsider!r} was judged as a recognised role"
    assert "does not recognise" in reason, reason
    assert repr(outsider) in reason, (
        "the refusal must name what it was given, or a caller cannot find the mistake"
    )
