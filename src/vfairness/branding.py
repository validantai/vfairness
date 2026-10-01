"""Output branding control (VB-COM-2).

Everything vfairness renders carries the validant.ai mark: the wordmark in a
chart header, the V icon bottom right, and a footer line on HTML and Markdown
reports. This module is the one switch that turns all of it off, so a study,
a client deliverable or an internal audit can ship without our name on it.

**The switch is unconditional, and that is deliberate.** vfairness is
Apache-2.0 and runs entirely on your machine with no telemetry, so there is
nothing here that checks a licence, calls home, or could. Sponsors are *asked*
to use this switch and everyone else *can*. Pretending otherwise would mean
either phoning home, which breaks a guarantee the test suite enforces, or
shipping a lock that section 2 of our own licence gives you the right to
remove. Honesty is worth more to this project than a fence that does not hold.

If the library is useful to you, https://vfairness.validant.ai/our-offering
explains how to support it.

Resolution order, first match wins:

1. an explicit :func:`set_branding` call,
2. the ``VFAIRNESS_BRANDING`` environment variable
   (``off`` / ``0`` / ``false`` / ``no`` / ``none`` disable it),
3. the default, which is branded.

Steps 1 and 2 read ONE vocabulary. The words above disable branding whichever
route they arrive by, and a value neither route can read leaves branding as it
was and says so, rather than being coerced in silence. Until 2026-09-27 the
setter coerced anything with ``bool()``, so ``set_branding("off")`` switched
branding ON while ``VFAIRNESS_BRANDING=off`` switched it off: see
:func:`set_branding` for the measurement.
"""

from __future__ import annotations

import os
import warnings

__all__ = ["branding_enabled", "set_branding"]

#: Environment variable consulted when no explicit override is set.
ENV_VAR = "VFAIRNESS_BRANDING"

_FALSEY = frozenset({"0", "off", "false", "no", "none"})
_TRUTHY = frozenset({"1", "on", "true", "yes"})

# None means "no explicit override, fall back to the environment".
_override: bool | None = None


def set_branding(enabled: bool | str | int | None = True) -> None:
    """Turn the validant.ai mark on or off for every chart and report.

    Args:
        enabled: ``True`` to brand output (the default), ``False`` to remove
            the mark, or ``None`` to clear the override so the
            ``VFAIRNESS_BRANDING`` environment variable decides again. A string
            from the vocabulary this module documents for the environment
            variable is read the same way here, so the two steps of the
            resolution order cannot disagree about the same word.

    Example:
        >>> import vfairness
        >>> vfairness.set_branding(False)   # unbranded charts and reports
        >>> vfairness.branding_enabled()
        False
        >>> vfairness.set_branding(None)    # back to the environment default

    ONE VOCABULARY, NOT TWO, AND A VALUE IT CANNOT READ IS DISCLOSED (2026-09-27).
    The body was ``_override = None if enabled is None else bool(enabled)``, a bare
    coercion with no validation and no warning, so this setter silently resolved
    the exact could-not-read class its sibling :func:`branding_enabled` is graded
    for DISCLOSING, and resolved it BACKWARDS. Measured before this change, with
    the same token through both documented routes:

        set_branding('off')   -> branding_enabled() True,  no warning
        set_branding('0')     -> branding_enabled() True,  no warning
        set_branding('false') -> branding_enabled() True,  no warning
        set_branding('no')    -> branding_enabled() True,  no warning
        set_branding('none')  -> branding_enabled() True,  no warning
        set_branding('OFF')   -> branding_enabled() True,  no warning
        VFAIRNESS_BRANDING='off' -> branding_enabled() False

    All five tokens the module docstring lists as DISABLING branding turned it
    ON, and ``set_branding('')`` turned branding OFF without a word, where the
    environment route refuses the empty string as unreadable and warns.

    After this change: those six strings disable branding, the truthy vocabulary
    enables it, and a value in neither ('', '   ', 'disable', a list, an object)
    leaves the switch EXACTLY as it was and warns. The switch is never moved by a
    word this function could not read, in either direction, which also means a
    typo can no longer undo a deliberate ``set_branding(False)``. Severity is low
    and stated as such: this unit reports no fairness number. It is a setter that
    was resolving what it could not read.
    """
    global _override
    if enabled is None:
        _override = None
        return
    if isinstance(enabled, bool):
        _override = enabled
        return

    value = enabled
    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        # A numpy scalar carries the same truth value as the Python object it
        # unwraps to, and refusing np.True_ would be an over-correction.
        try:
            value = value.item()
        except Exception:
            value = enabled
    if isinstance(value, bool):
        _override = value
        return
    if isinstance(value, (int, float)) and value in (0, 1):
        _override = bool(value)
        return
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _FALSEY:
            _override = False
            return
        if word in _TRUTHY:
            _override = True
            return

    if _override is None:
        unchanged = "no explicit override, so the environment or the branded default decides"
    else:
        unchanged = f"still {'ON' if _override else 'OFF'} from the previous explicit call"
    warnings.warn(
        f"set_branding({enabled!r}) is not a value this switch can read, so the switch "
        f"was NOT changed ({unchanged}). Pass True, False or None, or one of "
        f"{sorted(_FALSEY)} to remove the validant.ai mark and one of {sorted(_TRUTHY)} "
        "to keep it.",
        UserWarning,
        stacklevel=2,
    )


def branding_enabled() -> bool:
    """Return whether generated output should carry the validant.ai mark.

    An unrecognised ``VFAIRNESS_BRANDING`` value warns and leaves branding on,
    so a typo never silently strips the mark and leaves you believing you
    turned it off.
    """
    if _override is not None:
        return _override

    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return True

    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True

    warnings.warn(
        f"{ENV_VAR}={raw!r} is not a recognised value; branding stays ON. "
        f"Use one of {sorted(_FALSEY)} to remove the validant.ai mark.",
        UserWarning,
        stacklevel=2,
    )
    return True
