"""Public exception hierarchy for vfairness (VB-API-5).

Every error the library raises for a caller derives from :class:`VfairnessError`,
so downstream code can catch all of them with a single ``except VfairnessError``.
Each specific error also subclasses the built-in it historically replaced
(usually :class:`ValueError`), so existing ``except ValueError`` handlers keep
working unchanged. This lets the hierarchy be introduced without breaking any
current caller.

Example::

    from vfairness import FairnessAnalyzer, VfairnessError
    try:
        FairnessAnalyzer(task_type="classification").compute_all_metrics(...)
    except VfairnessError as err:
        # one place to handle any vfairness input/config problem
        log.warning("fairness analysis could not run: %s", err)
"""

from __future__ import annotations

__all__ = [
    "VfairnessError",
    "InvalidDataError",
    "InsufficientDataError",
    "ProtectedAttributeError",
    "ConfigurationError",
]


class VfairnessError(Exception):
    """Base class for every error vfairness raises for a caller."""


class InvalidDataError(VfairnessError, ValueError):
    """Input data is malformed or inconsistent.

    Raised for wrong shapes, mismatched array lengths, non-binary labels where
    binary is required, values outside an expected range, and similar problems
    with the data passed in.
    """


class InsufficientDataError(VfairnessError, ValueError):
    """Not enough data to compute the requested quantity.

    Distinct from an "insufficient evidence" verdict on a small group (which is
    a reported result, not an error): this is raised when a computation cannot
    proceed at all, for example an empty dataset.
    """


class ProtectedAttributeError(VfairnessError, ValueError):
    """A problem with the protected / sensitive attribute column.

    Raised when the protected attribute is missing, has only a single value
    (no groups to compare), or has cardinality too high to analyse meaningfully.
    """


class ConfigurationError(VfairnessError, ValueError):
    """Invalid configuration or parameter combination.

    Raised for unknown options, incompatible parameters, or a requested feature
    that needs an optional extra that is not installed.
    """
