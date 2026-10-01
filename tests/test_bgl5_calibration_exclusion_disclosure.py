"""The calibration exclusion disclosure must fire, and only when there is one.

WHY. ``_disclose_excluded_groups`` reads ``get_invalid_groups()`` as a DROP LIST, so
an empty list there means "none were dropped". That reading is correct, including
when the threshold cannot flag anything.

``GroupManager.get_invalid_groups`` gained a disclosure on 2026-09-27 for a
``min_group_size`` of 1 or less, because every group is built from levels that
actually occur, so no group can hold fewer than one row and none can ever be too
small: the empty answer is vacuous for a caller that reads it as a clean bill. But
``group_calibrator.py`` builds its manager with ``min_group_size=1`` DELIBERATELY,
to include every present group in the intervention view, and it reaches this helper.
The disclosure fired on a fully covered run and reddened a control whose whole
subject is that a complete analysis says nothing.

The early return closes that. This file exists because SABOTAGING it the other way
left the suite green: replacing the whole helper with ``return []`` suppressed every
exclusion disclosure and nothing noticed. A guard whose over-correction is invisible
is half a guard, so both directions are pinned here with real numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.post_processing.calibration.metrics import expected_calibration_error


def _two_groups(n_small: int):
    """100 rows of "a" and ``n_small`` of "b", with a real calibration signal."""
    rng = np.random.default_rng(7)
    n_a = 100
    y_true = np.concatenate([rng.integers(0, 2, n_a), rng.integers(0, 2, n_small)])
    y_prob = np.concatenate([rng.uniform(0.4, 0.6, n_a), rng.uniform(0.0, 1.0, n_small)])
    groups = np.array(["a"] * n_a + ["b"] * n_small)
    return y_true, y_prob, groups


def _run(min_group_size: int, n_small: int):
    y, p, g = _two_groups(n_small)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = expected_calibration_error(y, p, protected_attr=g, min_group_size=min_group_size)
    return result, [str(w.message) for w in caught]


def test_a_real_exclusion_is_disclosed_and_names_the_group():
    """THE OVER-CORRECTION CONTROL, and the one that was missing.

    A group of 5 rows against a floor of 30 is genuinely dropped, and the reader
    has to be told, or the group values are a subset presented as the whole.
    """
    result, messages = _run(min_group_size=30, n_small=5)
    text = " ".join(messages)
    assert "excluded" in text.lower(), messages
    assert "'b'" in text or '"b"' in text or "b" in text, messages
    assert "NOT assessed" in text, messages
    assert "b" not in (result.group_values or {}), result.group_values
    assert "a" in (result.group_values or {}), result.group_values


def test_a_disabled_threshold_discloses_nothing_because_nothing_was_dropped():
    """The fix. min_group_size=1 cannot flag any group, so no group was dropped.

    group_calibrator passes 1 on purpose, to keep every present group in the
    intervention view. Reporting an exclusion there would name groups that are all
    still in the analysis.
    """
    result, messages = _run(min_group_size=1, n_small=5)
    assert not [m for m in messages if "excluded" in m.lower()], messages
    assert not [m for m in messages if "vacuous" in m.lower()], messages
    assert {"a", "b"} <= set(result.group_values or {}), result.group_values


@pytest.mark.parametrize("floor", [2, 6, 30])
def test_the_boundary_is_the_group_size_and_not_the_threshold_alone(floor):
    """A group of 5 is dropped at 6 and at 30, and kept at 2.

    This is what makes the early return a statement about VACUITY rather than a
    blanket suppression: every threshold that can flag something still does.
    """
    result, messages = _run(min_group_size=floor, n_small=5)
    dropped = "b" not in (result.group_values or {})
    disclosed = bool([m for m in messages if "excluded" in m.lower()])
    assert dropped == (floor > 5), (floor, result.group_values)
    assert disclosed == dropped, (floor, messages)
