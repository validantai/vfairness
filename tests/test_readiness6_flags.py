"""READINESS-6: ten coercers turned a yes/no FLAG into a measurement.

Each one reads a reported quantity for a chart, a table or a report and answers
None (or NaN) when nothing was reported. Each one's docstring says it refuses
"anything that is not a number". None of them refused a bool, and ``float(True)``
is 1.0: a finite number that passes every check they make.

1.0 is not a harmless value in any of these places. On a drift chart it is the
maximum, drawn as a full-width bar. On the 1-5 risk register it is a severity. On
a correlation heatmap it is a perfect correlation. On a proportion-mediated field
it is the whole effect.

``np.bool_`` is the half that gets missed everywhere, because it is NOT a Python
bool, so ``isinstance(value, bool)`` is False for it and a boolean column read
out of a DataFrame walks straight through the guard that was written to stop
exactly this.

The refusal lives in ``_triage.is_flag`` rather than in ten copies, because ten
copies of a predicate is how six mutually disagreeing versions of ``is_measured``
came to exist in the first place (see test_readiness6_measured.py).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vfairness._triage import is_flag
from vfairness.operations.pulse.agent_probe import _finite_or_none as probe_finite
from vfairness.preprocessing.feature_engineering.visualization import _finite_or_nan
from vfairness.rendering.adapters_experimentation import _finite as experimentation_finite
from vfairness.rendering.adapters_fairness import _is_finite, _to_finite_float
from vfairness.rendering.adapters_feature_engineering import _finite_corr, _finite_number
from vfairness.rendering.adapters_monitoring import _measured as monitoring_measured
from vfairness.rendering.adapters_regression import _finite as regression_finite
from vfairness.rendering.adapters_robustness import _finite_or_none as robustness_finite

FLAGS = [
    ("python True", True),
    ("python False", False),
    ("numpy True", np.bool_(True)),
    ("numpy False", np.bool_(False)),
]

# (label, callable, the value that means "nothing was reported")
COERCERS = [
    ("adapters_fairness._to_finite_float", _to_finite_float, None),
    ("adapters_regression._finite", regression_finite, None),
    ("adapters_experimentation._finite", experimentation_finite, None),
    ("adapters_robustness._finite_or_none", robustness_finite, None),
    ("adapters_feature_engineering._finite_number", _finite_number, None),
    ("adapters_feature_engineering._finite_corr", _finite_corr, None),
    ("adapters_monitoring._measured", monitoring_measured, None),
    ("agent_probe._finite_or_none", probe_finite, None),
]


@pytest.mark.parametrize("name,fn,absent", COERCERS)
@pytest.mark.parametrize("flag_label,flag", FLAGS)
def test_a_flag_is_not_a_reported_quantity(name, fn, absent, flag_label, flag):
    got = fn(flag)
    assert got is absent, f"{name} read {flag_label} as {got!r}"


@pytest.mark.parametrize("flag_label,flag", FLAGS)
def test_the_bool_returning_and_nan_returning_ones_too(flag_label, flag):
    """Two of the ten do not return None: one returns a bool and one returns
    NaN. Same defect, different spelling of "not measured"."""
    assert _is_finite(flag) is False, flag_label
    assert math.isnan(_finite_or_nan(flag)), flag_label


@pytest.mark.parametrize("name,fn,absent", COERCERS)
def test_real_measurements_still_come_through(name, fn, absent):
    """OVER-CORRECTION CONTROL, the whole point of this file's twin.

    A guard that refuses flags by refusing everything is not a fix, and would be
    the harder defect to see: the aggregate would drop real numbers and then
    report honestly on what was left.
    """
    for value in (0.42, 0, 1, -3.2, np.float64(0.42), np.float32(0.42), np.int64(5)):
        assert fn(value) is not absent, f"{name} discarded a real {type(value).__name__}"
    assert fn(0.0) is not absent, f"{name} discarded a measured ZERO"


def test_a_measured_zero_survives_and_is_not_confused_with_absence():
    """The distinction these functions exist for. 0.0 is falsy in Python, so a
    guard written with `if not value` would drop it, and a measured zero is the
    most consequential reading in several of these surfaces (a group with zero
    selections, a correlation of exactly zero, no drift at all)."""
    for name, fn, absent in COERCERS:
        got = fn(0.0)
        assert got is not absent, f"{name} lost a measured zero"
        assert got == 0.0, f"{name} changed a measured zero to {got!r}"
    assert _is_finite(0.0) is True
    assert _finite_or_nan(0.0) == 0.0


def test_the_probe_coercer_returns_null_instead_of_raising():
    """`agent_probe._finite_or_none` was annotated `x: float` and called
    `math.isfinite(x)` directly, so anything else RAISED where the function's
    whole purpose is to produce a null at a JSON boundary. An annotation is not
    a check, and a JSON boundary is exactly where a wrong-shaped value arrives.
    """
    for bad in (None, "0.5", [1], {}, object()):
        assert probe_finite(bad) is None, f"{bad!r} did not become null"
    assert probe_finite(float("nan")) is None
    assert probe_finite(float("inf")) is None
    assert probe_finite(0.42) == pytest.approx(0.42)


def test_is_flag_itself():
    assert is_flag(True) is True
    assert is_flag(False) is True
    assert is_flag(np.bool_(True)) is True
    assert is_flag(np.bool_(False)) is True
    # A number is not a flag, including the numbers a bool coerces to.
    assert is_flag(1) is False
    assert is_flag(0) is False
    assert is_flag(1.0) is False
    assert is_flag(np.int64(1)) is False
    assert is_flag(np.float64(1.0)) is False
    assert is_flag(None) is False
    assert is_flag("True") is False


def test_the_rendering_coercers_still_read_serialised_numbers():
    """OVER-CORRECTION CONTROL, and a deliberate contract difference.

    A numeric STRING stays acceptable in the rendering layer: those rows arrive
    from JSON and CSV, where "0.5" is a real measurement that was serialised, and
    refusing it would discard evidence. `compliance._as_float` keeps the opposite
    contract for the opposite and equally correct reason, that its input is a
    caller's free-text field where parsing would be a guess. Pinned so the
    divergence is a decision rather than an oversight.
    """
    from vfairness.operations.reporting.compliance import _as_float

    assert _to_finite_float("0.5") == pytest.approx(0.5)
    assert monitoring_measured("0.5") == pytest.approx(0.5)
    assert _as_float("0.5") is None
