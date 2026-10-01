"""`help(vfairness)` is one of only TWO documentation surfaces a pip user gets.

The wheel ships `/src` and nothing else: METHODOLOGY.md, BETA.md and
QUALITY_AND_HARDENING.md do not reach an installed package. So the package
docstring and the README carry the whole burden of teaching the contract that
matters most, and until 2026-09-07 the docstring taught none of it: 4,746
characters that never once said "NaN", "not_assessable", or "three".

A user who does not know the third state exists will write `value or 0.0` or
`np.nan_to_num(...)` and turn every could-not-check into a measured perfect
parity, which is precisely the defect class this library spent two days
removing from its own code. Teaching it is not documentation polish; it is the
difference between the contract holding and being silently undone downstream.
"""

import vfairness


def test_the_package_docstring_names_the_third_state():
    d = vfairness.__doc__ or ""
    assert "NaN" in d, "the docstring never mentions NaN, which IS the third state"
    assert "Three States" in d or "three states" in d.lower()


def test_it_says_what_nan_means_rather_than_only_naming_it():
    d = vfairness.__doc__ or ""
    assert "COMPARISON NEVER HAPPENED" in d.upper() or "never happened" in d.lower()


def test_it_warns_against_the_two_ways_users_erase_the_third_state():
    """The concrete failure, not an abstraction: `or 0.0` and `nan_to_num` are
    how a NaN becomes a fabricated all-clear in someone else's codebase."""
    d = vfairness.__doc__ or ""
    assert "or 0.0" in d
    assert "nan_to_num" in d


def test_it_points_at_where_the_same_state_surfaces_in_the_report_and_the_gates():
    d = vfairness.__doc__ or ""
    assert "not_assessable" in d
    assert "assert_fairness" in d or "ModelFairnessGate" in d
