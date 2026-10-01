"""READINESS-6: six implementations of "was this actually measured?", and they
disagreed on five of nine inputs.

`_triage.py`'s own docstring says it exists so this question is answered once.
It was answered six times, and the canonical answer was one of the wrong ones.

Measured 2026-09-10, before the fix::

                                 0.23  np.float32  np.int64  True   inf  '0.5'
    _triage.is_measured          True       False     False  False False  False
    explainer._measured          True        True      True  False False  False
    correspondence._is_measured  True        True      True   True False   True
    visualization._is_measured   True       False     False  False False  False
    integrations._is_measured    True        True      True  False False  False
    in_processing._measured      True       False     False  False  True  False

Two things in there matter more than the count.

`_triage.is_measured` and `visualization._is_measured` classified a REAL, FINITE
`np.float32` as NOT MEASURED, because the test was `isinstance(value, (int,
float))` and numpy scalars are not Python int or float subclasses. That is the
opposite of every other finding in this audit: a measurement reported as a
could-not-check. It reads as caution and is actually evidence discarded, which
is much harder to spot than a fabricated verdict, and it was in the module
nominated as canonical.

`in_processing._measured` accepted INFINITY, because its test was a bare NaN
check. `risk_ratio` returns `inf` for a group with zero observations and `inf`
compares greater than every threshold, so an unmeasurable group silently became
the worst breach in the report. That is rule 1 of `_triage`, and the module that
needed it most did not use it.

`correspondence._is_measured` is deliberately NOT delegated: its inputs are
outcome lists where a bool is a recorded yes/no outcome and a numeric string is
a legitimate cell. Its contract genuinely differs, and it says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics.integrations import _is_measured as integrations_
from vfairness.evaluation.vfairness_metrics.visualization import _is_measured as visualization_
from vfairness.explainer import _measured as explainer_
from vfairness.in_processing.analyzer import _measured as in_processing_

IMPLEMENTATIONS = {
    "_triage.is_measured": is_measured,
    "explainer._measured": explainer_,
    "visualization._is_measured": visualization_,
    "integrations._is_measured": integrations_,
    "in_processing._measured": in_processing_,
}

# (label, value, is it a measurement?)
CASES = [
    ("python float", 0.23, True),
    ("python int", 5, True),
    ("zero", 0.0, True),
    ("negative", -3.2, True),
    ("np.float32", np.float32(0.23), True),
    ("np.float64", np.float64(0.23), True),
    ("np.int64", np.int64(5), True),
    ("np.int32", np.int32(5), True),
    ("python bool", True, False),
    ("numpy bool", np.bool_(True), False),
    ("positive infinity", float("inf"), False),
    ("negative infinity", float("-inf"), False),
    ("numpy infinity", np.float32("inf"), False),
    ("nan", float("nan"), False),
    ("numpy nan", np.float32("nan"), False),
    ("None", None, False),
    ("numeric string", "0.5", False),
    ("list", [1], False),
    ("array", np.array([1]), False),
]


@pytest.mark.parametrize("name,fn", list(IMPLEMENTATIONS.items()))
@pytest.mark.parametrize("label,value,expected", CASES)
def test_every_implementation_gives_the_canonical_answer(name, fn, label, value, expected):
    assert bool(fn(value)) is expected, (
        f"{name} answered {bool(fn(value))!r} for {label}; the canonical answer is "
        f"{expected!r}. A value is measured only when it is a real, finite, "
        f"non-boolean number."
    )


def test_they_agree_with_each_other_on_every_case():
    """The property that actually matters, stated directly rather than inferred
    from the parametrised runs above."""
    for label, value, _ in CASES:
        answers = {name: bool(fn(value)) for name, fn in IMPLEMENTATIONS.items()}
        assert len(set(answers.values())) == 1, f"{label}: {answers}"


def test_a_finite_numpy_measurement_is_not_discarded():
    """The dangerous direction, called out on its own.

    Every other finding in this audit is a could-not-check reported as a
    measurement. This one is the reverse, and the reverse is harder to see: the
    aggregate drops a real number and then reports honestly on what is left.
    """
    for value in (np.float32(0.23), np.float64(0.23), np.int64(5), np.int32(5)):
        assert is_measured(value) is True, f"{value!r} ({type(value).__name__})"


def test_infinity_is_not_a_measurement():
    """risk_ratio returns inf for a group with zero observations, and inf beats
    every threshold, so an unmeasurable group becomes the worst breach."""
    for value in (float("inf"), float("-inf"), np.float32("inf")):
        assert is_measured(value) is False, f"{value!r}"


def test_correspondence_keeps_its_own_contract_deliberately():
    """OVER-CORRECTION CONTROL. Not every sibling should be folded in.

    `agents/correspondence._is_measured` takes outcome lists, where a bool is a
    recorded yes/no callback and a numeric string is a legitimate cell. Folding
    it into the numeric rule would reject real outcomes. It is left alone ON
    PURPOSE, and this test exists so that is a decision rather than an omission.
    """
    from vfairness.agents.correspondence import _is_measured as correspondence_

    assert correspondence_(True) is True, "a recorded yes/no outcome was rejected"
    assert correspondence_(0.5) is True
    # And it still refuses what nobody recorded, which is its actual job.
    assert correspondence_(None) is False
    assert correspondence_(float("nan")) is False
    assert correspondence_(np.float32("nan")) is False


def test_the_assumption_that_makes_the_numpy_bool_exclusion_redundant():
    """`_triage.is_measured` names `np.bool_` in its first gate, and that gate is
    REDUNDANT: measured on numpy 2.4.6, `np.bool_` is not registered as
    `numbers.Real`, so the second gate rejects it anyway. A sabotage removing the
    exclusion stayed green, which is how the redundancy was found.

    The exclusion is kept as defence in depth, and this test pins the assumption
    underneath it. If a future numpy registers its boolean as a real number, the
    second gate stops being sufficient on its own, and a recorded yes/no would
    read as a 1.0 measurement everywhere `is_measured` is the only filter. This
    test going red is the warning that the first gate has become load-bearing.
    """
    import numbers

    assert not isinstance(np.bool_(True), numbers.Real), (
        "numpy now registers np.bool_ as numbers.Real. The explicit np.bool_ "
        "exclusion in _triage.is_measured is now LOAD-BEARING rather than "
        "redundant; do not remove it, and re-check every other predicate that "
        "filters on numbers.Real alone."
    )
    # And either way, the behaviour that actually matters holds.
    assert is_measured(np.bool_(True)) is False
    assert is_measured(np.bool_(False)) is False
