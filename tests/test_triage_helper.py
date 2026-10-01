"""`vfairness._triage`, the one place "was this measured?" is answered.

Phase 0.1 of the fifth-iteration remediation plan. Each test here pins a rule
that a real finding depended on, so the cases are not hypothetical.
"""

from __future__ import annotations

import math

import pytest

from vfairness._triage import (
    describe_unmeasured,
    is_measured,
    measured_values,
    partition_measured,
)


@pytest.mark.parametrize("value", [0.0, -3.2, 1, 1e-12, -0.0, 1e308])
def test_real_finite_numbers_are_measured(value):
    assert is_measured(value)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_absent_and_non_finite_values_are_not_measured(value):
    assert not is_measured(value)


def test_infinity_is_not_measured_because_it_outranks_every_threshold():
    """H-04. `risk_ratio` returns inf when a group has zero observations.

    inf compares greater than any threshold, so an unmeasurable group silently
    becomes the WORST breach in a report rather than an absent one.
    """
    assert not is_measured(float("inf"))
    assert float("inf") > 0.8  # the comparison that made it a fabricated breach


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_is_not_a_measurement(value):
    """C-08 and the groundedness guard. bool is a subclass of int and
    float(True) == 1.0, so a rung answering True clamps to a PERFECT score."""
    assert not is_measured(value)
    assert float(True) == 1.0  # the coercion that made it a fabricated all-clear


@pytest.mark.parametrize("value", ["0.5", [1.0], {"a": 1}, object()])
def test_non_numbers_are_not_measured(value):
    assert not is_measured(value)


def test_partition_returns_both_halves():
    measured, unmeasured = partition_measured(
        {"dp": 0.1, "eo": float("nan"), "pp": None, "di": 0.9}
    )
    assert measured == {"dp": 0.1, "di": 0.9}
    assert unmeasured == ["eo", "pp"]


def test_partition_keeps_a_measured_zero():
    """A measured 0.0 is a real result: perfect parity. It must not be confused
    with an absent one, which is the confusion this whole module is about."""
    measured, unmeasured = partition_measured({"dp": 0.0})
    assert measured == {"dp": 0.0}
    assert unmeasured == []


def test_unmeasured_names_are_sorted_for_stable_wording():
    _, unmeasured = partition_measured({"z": None, "a": float("nan"), "m": None})
    assert unmeasured == ["a", "m", "z"]


def test_partition_of_an_empty_mapping_measures_nothing():
    assert partition_measured({}) == ({}, [])


def test_measured_values_drops_the_rest():
    assert measured_values([1.0, float("nan"), None, 2.0, True]) == [1.0, 2.0]


def test_describe_is_empty_when_everything_was_measured():
    """So a caller can append it unconditionally and a clean run reads unchanged."""
    assert describe_unmeasured([]) == ""


def test_describe_names_what_was_missed():
    text = describe_unmeasured(["equal_opportunity", "predictive_parity"], "metric")
    assert "2 metrics could not be measured" in text
    assert "equal_opportunity" in text and "predictive_parity" in text
    assert "excluded from this verdict" in text


def test_describe_uses_the_singular_for_one():
    assert describe_unmeasured(["eo"], "metric").startswith("1 metric could not be measured")


def test_the_helper_never_substitutes_a_value():
    """The rule that separates this from the defect it replaces.

    Filling in a default IS the defect: it fabricates the thing the caller asked
    to have measured. The partition hands back both halves and nothing else.
    """
    measured, unmeasured = partition_measured({"a": float("nan")})
    assert measured == {}
    assert unmeasured == ["a"]
    assert not any(v == 0.0 for v in measured.values())
    assert not math.isnan(sum(measured.values()) if measured else 0.0)
