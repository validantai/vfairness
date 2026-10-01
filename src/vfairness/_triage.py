"""Tell measured values apart from unmeasurable ones, in one place.

WHY THIS MODULE EXISTS. The fifth-iteration audit (2026-09-07) found 9 CRITICAL
and 15 HIGH instances of a single mechanism:

    The measurement layer is already honest. The layer above it discards the
    honesty.

The metrics return NaN and warn. Then ``scan_fairness_violations`` drops them,
``recommend_calibration_strategy`` keeps only the 0.0, ``representation_severity``
grades an explicit refusal "pass", and ``.get(dim, 5)`` fills in an absent judge
score with the neutral value. In every case the aggregate reads as a measured
result, and a reader cannot tell it apart from one.

The failure is not that anybody wrote ``if math.isnan(x)`` wrongly. It is that
each aggregate re-invented the question "did this actually get measured?", and a
handful of them answered it with ``or 0.0``. This module answers it once.

TWO RULES THIS ENCODES.

1. **A value is measured only if it is a real, finite number.** ``None``, NaN and
   the infinities are all "could not check". Infinity matters more than it looks:
   ``risk_ratio`` returns ``inf`` when a group has zero observations, and ``inf``
   compares greater than every threshold, so an unmeasurable group silently
   becomes the worst breach in the report.

2. **A bool is NOT a measurement.** ``bool`` is a subclass of ``int`` and
   ``float(True) == 1.0``, so a rung answering ``True`` instead of a score
   clamps to a perfect 1.0. ``validity/groundedness.py`` already rejects booleans
   first for exactly this reason; the rule belongs here rather than in one module.

WHAT THIS DELIBERATELY DOES NOT DO. It does not substitute a value. Filling in a
default is the defect being removed: it fabricates the thing the caller asked to
have measured. Callers get the partition and must report the unmeasured half.
"""

from __future__ import annotations

import decimal
import math
import numbers
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Tuple

import numpy as np

# The types that CAN carry a measurement. ``numbers.Real`` covers Python floats
# and ints, Fraction, and the numpy scalars (numpy registers them), and it
# excludes str, list, complex and ndarray.
#
# ``decimal.Decimal`` has to be named beside it, and unlike the ``np.bool_``
# exclusion in :func:`is_measured` this one IS load-bearing. The stdlib
# deliberately registers Decimal with ``numbers.Number`` and NOT with
# ``numbers.Real``, so measured 2026-09-27 on this repo,
# ``isinstance(Decimal("0.5"), numbers.Real)`` is False and the canonical rule
# answered ``is_measured(Decimal("0.5")) -> False`` and
# ``unmeasurable_reason(Decimal("0.5")) -> 'not a number (Decimal)'`` for a real,
# finite number: the READINESS-6 numpy-scalar defect over again, a measurement
# reported as a could-not-check. Two other coercers in this same repo already
# accept it on purpose (``agents/correspondence._is_measured``, whose comment
# names Decimal, and ``operations/pulse/regulatory._as_real``, whose fallback
# names it), so the module that exists to END six disagreeing copies of this
# rule was the one copy that disagreed. A Decimal reaches a metric from a
# database NUMERIC column and from ``json.loads(parse_float=Decimal)``.
#
# Decimal does NOT weaken either rule: ``Decimal("NaN")`` and
# ``Decimal("Infinity")`` still fail the finiteness gate below.
_MEASURABLE_TYPES = (numbers.Real, decimal.Decimal)

__all__ = [
    "is_measured",
    "partition_measured",
    "measured_values",
    "describe_unmeasured",
    "unmeasurable_reason",
    "is_flag",
]


def is_measured(value: Any) -> bool:
    """True only for a real, finite number that is not a bool.

    >>> is_measured(0.0), is_measured(-3.2)
    (True, True)
    >>> is_measured(None), is_measured(float("nan")), is_measured(float("inf"))
    (False, False, False)
    >>> is_measured(True)          # bool is an int; float(True) == 1.0
    False

    READINESS-6, 2026-09-10. The test was ``isinstance(value, (int, float))``,
    which is False for ``np.float32`` and ``np.int64``: numpy scalars are not
    Python int or float subclasses (``np.float64`` happens to be, the others are
    not). So a REAL, FINITE numpy measurement was classified NOT MEASURED by the
    module the docstring above nominates as the single answer, contradicting its
    own rule 1.

    That is the dangerous direction and the one this audit had not been guarding
    against. Every other finding here was a could-not-check reported as a
    measurement; this is a measurement reported as a could-not-check, which
    reads as caution while actually discarding evidence. An aggregate built on
    it drops real numbers and then says so honestly, which is much harder to
    spot than a fabricated verdict.

    ``numbers.Real`` is the right test: numpy registers its scalar types with
    it, and it excludes str, list, complex and ndarray. ``decimal.Decimal`` is
    named beside it because the stdlib does NOT register it as a Real and a
    Decimal is a real, finite number; see ``_MEASURABLE_TYPES`` above for the
    measurement.

    ``np.bool_`` is named beside ``bool`` as DEFENCE IN DEPTH, not because it is
    load-bearing today. Measured on numpy 2.4.6,
    ``isinstance(np.bool_(True), numbers.Real)`` is False, so the second gate
    already rejects it and this first one is redundant for that type. An earlier
    draft of this docstring asserted the opposite, that numpy's boolean "would
    otherwise pass rule 2"; a sabotage run removing the exclusion stayed GREEN
    and showed the claim was false. It is kept because the redundancy costs
    nothing and a future numpy that DID register its boolean as a real number
    would otherwise silently turn every recorded yes/no into a 1.0 measurement.
    That assumption is pinned in ``tests/test_readiness6_measured.py`` so the
    day it stops holding is a red test rather than a silent change of meaning.

    >>> import numpy as np
    >>> is_measured(np.float32(0.23)), is_measured(np.int64(5))
    (True, True)
    >>> is_measured(np.bool_(True)), is_measured(np.float32("inf"))
    (False, False)
    >>> from decimal import Decimal
    >>> is_measured(Decimal("0.5")), is_measured(Decimal("NaN"))
    (True, False)
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, _MEASURABLE_TYPES):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def partition_measured(
    values: Mapping[Hashable, Any],
) -> Tuple[Dict[Hashable, float], List[Hashable]]:
    """Split a name -> value mapping into what was measured and what was not.

    Returns ``(measured, unmeasured)`` where ``measured`` maps each name to a
    finite float and ``unmeasured`` is the sorted list of names that could not be
    measured. Sorted so a report's wording is stable across runs.

    The point of returning BOTH halves is that an aggregate cannot then quietly
    keep only the first one, which is the shape of nearly every finding this
    module was written for.

    >>> partition_measured({"a": 0.1, "b": float("nan"), "c": None})
    ({'a': 0.1}, ['b', 'c'])
    """
    measured: Dict[Hashable, float] = {}
    unmeasured: List[Hashable] = []
    for name, value in values.items():
        if is_measured(value):
            measured[name] = float(value)
        else:
            unmeasured.append(name)
    return measured, sorted(unmeasured, key=str)


def measured_values(values: Iterable[Any]) -> List[float]:
    """The finite numbers from an iterable, dropping the rest.

    Use this ONLY where the count of dropped items is reported separately; on its
    own it is the silent drop this module exists to prevent.
    """
    return [float(v) for v in values if is_measured(v)]


def describe_unmeasured(unmeasured: List[Hashable], noun: str = "metric") -> str:
    """One clause naming what could not be measured, or "" when nothing was.

    Returning "" for the clean case is deliberate: a caller can append this
    unconditionally, and a fully measured run reads exactly as it did before.

    >>> describe_unmeasured([])
    ''
    >>> describe_unmeasured(["equal_opportunity"], "metric")
    '1 metric could not be measured and is excluded from this verdict: equal_opportunity'
    """
    if not unmeasured:
        return ""
    plural = noun if len(unmeasured) == 1 else f"{noun}s"
    names = ", ".join(str(u) for u in unmeasured)
    return (
        f"{len(unmeasured)} {plural} could not be measured and "
        f"{'is' if len(unmeasured) == 1 else 'are'} excluded from this verdict: {names}"
    )


def unmeasurable_reason(value: Any) -> str:
    """Name WHY a value is not a measurement, in one clause for an operator.

    The reason matters as much as the refusal, because the two sentinels reach a
    metric by different routes and an operator fixes them differently. NaN comes
    from a computation that had nothing to work with. Infinity comes from a
    division whose denominator was zero, which on the ratio family means a group
    with no selections AT ALL: the most extreme unfairness the data can express,
    arriving as a value that no comparison can grade.

    >>> unmeasurable_reason(float("nan"))
    'NaN: insufficient evidence'
    >>> unmeasurable_reason(float("inf"))
    'infinite: no comparison can grade it'
    >>> unmeasurable_reason(None)
    'no value was recorded'
    >>> unmeasurable_reason(0.5)
    ''
    >>> from decimal import Decimal
    >>> unmeasurable_reason(Decimal("0.5")), unmeasurable_reason(Decimal("NaN"))
    ('', 'NaN: insufficient evidence')

    READINESS-6, 2026-09-10. Written because four sites guarded NaN and let
    infinity through, each with a comment explaining at length why an ungraded
    value must not fall into a PASS. Measured on this repo before the fix:
    ``ModelFairnessGate(metrics=["disparate_impact_ratio"],
    thresholds={"disparate_impact_ratio": 0.8}).evaluate_from_metrics(
    {"disparate_impact_ratio": float("inf")})`` returned status APPROVED,
    ``approved=True``, ``passed=True``, an EMPTY message and a GitHub check
    conclusion of ``success``, indistinguishable from the genuine pass at 0.95,
    while the NaN beside it failed closed and explained itself.
    """
    if value is None:
        return "no value was recorded"
    if isinstance(value, (bool, np.bool_)):
        return "a yes/no flag, not a measurement"
    if not isinstance(value, _MEASURABLE_TYPES):
        return f"not a number ({type(value).__name__})"
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        # A number that will not read as one, e.g. Decimal('sNaN'), whose
        # float() raises. :func:`is_measured` already refuses it through the
        # same conversion; without this clause the exception escaped from the
        # function whose whole job is to NAME the refusal.
        return f"not readable as a number ({type(value).__name__})"
    if math.isnan(as_float):
        return "NaN: insufficient evidence"
    if math.isinf(as_float):
        return "infinite: no comparison can grade it"
    return ""


def is_flag(value: Any) -> bool:
    """True for a boolean, Python's or numpy's. A flag is never a measurement.

    Separate from :func:`is_measured` because a dozen coercers in the rendering
    layer deliberately accept a numeric STRING (their rows arrive from JSON and
    CSV, where "0.5" is a real measurement that was serialised) while still
    needing to refuse a flag. They cannot use the full canonical rule without
    discarding real data, and they should not each carry their own bool test:
    that is precisely how six mutually disagreeing copies of ``is_measured``
    came to exist.

    ``np.bool_`` is the half that gets missed. It is NOT a Python bool, so
    ``isinstance(value, bool)`` is False for it, and a boolean column read out of
    a DataFrame walks past the guard and becomes 1.0. On a drift chart 1.0 is
    the maximum; on a 1-5 risk register it is a severity; on a correlation
    heatmap it is a perfect correlation.

    >>> is_flag(True), is_flag(False)
    (True, True)
    >>> import numpy as np
    >>> is_flag(np.bool_(True))
    True
    >>> is_flag(1), is_flag(1.0), is_flag("True"), is_flag(None)
    (False, False, False, False)
    """
    return isinstance(value, (bool, np.bool_))
