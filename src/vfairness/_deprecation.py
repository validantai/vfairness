"""Deprecation helper (VB-API-2).

A single, consistent way to mark a public symbol as deprecated. It emits a
``DeprecationWarning`` naming the replacement and the version in which the
symbol will be removed, so callers get a one-minor-version warning window as
promised by the deprecation policy in ``docs/API_STABILITY.md``.

Usage::

    from vfairness._deprecation import deprecated

    @deprecated("use compute_all_metrics()", removed_in="0.1.0")
    def old_compute(...):
        ...
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(reason: str, *, removed_in: Optional[str] = None) -> Callable[[F], F]:
    """Mark a callable as deprecated.

    Args:
        reason: What to use instead, in plain words.
        removed_in: The version the symbol is scheduled to be removed in.

    Returns:
        A decorator that warns once per call site group and then delegates.
    """

    def decorator(func: F) -> F:
        message = f"{func.__name__} is deprecated: {reason}"
        if removed_in:
            message += f" It is scheduled for removal in vfairness {removed_in}."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = message  # type: ignore[attr-defined]  # discoverable marker on a plain function object
        return wrapper  # type: ignore[return-value]

    return decorator


def warn_deprecated(name: str, reason: str, *, removed_in: Optional[str] = None) -> None:
    """Emit a deprecation warning imperatively (for non-callable aliases)."""
    message = f"{name} is deprecated: {reason}"
    if removed_in:
        message += f" It is scheduled for removal in vfairness {removed_in}."
    warnings.warn(message, DeprecationWarning, stacklevel=2)
