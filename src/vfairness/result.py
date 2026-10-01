"""Library-wide structured task-result envelope.

Every task handler in :mod:`vfairness.operations` returns a small JSON
envelope to its consumer. Historically that shape was written by hand at
each return site as ``{"success": True, "data": ...}`` or
``{"success": False, "error": ...}``. :class:`TaskResult` formalizes that
shape in one place so the contract is versioned and testable, while
keeping the serialized output backward compatible: :meth:`TaskResult.to_dict`
still emits ``success`` plus ``data`` / ``error`` exactly as before, and
only adds two additive keys (``schema_version`` and ``task_type``) that
existing key-based consumers ignore.

Usage::

    from vfairness.result import TaskResult

    return TaskResult.ok("vfairness_data_validation", data=report).to_dict()
    return TaskResult.fail("vfairness_pulse_run", "no artifact provided").to_dict()

The dataclass is stdlib-only (no third-party dependency) so it is safe to
import from any handler regardless of the installed extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Bumped only on a breaking change to the envelope shape. Consumers may read
# this to negotiate behaviour across library versions.
SCHEMA_VERSION = "1.0"


def _coerce_success(raw: Any) -> Optional[bool]:
    """The reported success flag, or None when it is not a boolean.

    Stdlib only, so this module stays importable from any handler regardless of
    the installed extras: ``numpy.bool_`` is unwrapped through its ``.item()``
    rather than by importing numpy. Only real booleans and the ints 0 and 1 are
    accepted; everything else, including a string and a NaN, is "no verdict",
    which the caller turns into a stated failure.
    """
    if isinstance(raw, bool):
        return raw
    item = getattr(raw, "item", None)  # numpy.bool_ and friends
    if callable(item):
        try:
            unwrapped = item()
        except Exception:  # noqa: BLE001 -- an exotic object is simply not a flag
            return None
        if isinstance(unwrapped, bool):
            return unwrapped
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    return None


@dataclass
class TaskResult:
    """A structured result envelope for a vfairness task handler.

    Fields
    ------
    task_type:
        The dispatch key the consumer invoked, e.g. ``"vfairness_pulse_run"``.
    success:
        Whether the task completed successfully.
    schema_version:
        Envelope contract version. Defaults to :data:`SCHEMA_VERSION`.
    data:
        The successful result payload (handler specific). ``None`` on failure.
    error:
        A human-readable error message. ``None`` on success.
    warnings:
        Optional non-fatal notes surfaced alongside a result.
    """

    task_type: str
    success: bool
    schema_version: str = SCHEMA_VERSION
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[Any]] = None

    # Constructors

    @classmethod
    def ok(
        cls,
        task_type: str,
        data: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[Any]] = None,
    ) -> "TaskResult":
        """Build a successful result."""
        return cls(task_type=task_type, success=True, data=data, warnings=warnings)

    @classmethod
    def fail(
        cls,
        task_type: str,
        error: str,
        warnings: Optional[List[Any]] = None,
    ) -> "TaskResult":
        """Build a failed result."""
        return cls(task_type=task_type, success=False, error=error, warnings=warnings)

    @classmethod
    def from_envelope(cls, task_type: str, envelope: Any) -> "TaskResult":
        """Adopt an existing ``{"success", "data"|"error"}`` dict.

        Used to formalize the output of a subsystem (for example the Pulse
        orchestrator) that already returns the informal envelope, without
        re-nesting its ``data`` or dropping any of its keys.

        A ``success`` flag that is not a boolean is adopted as a FAILURE and the
        reason is stated. G13, 2026-09-30: this was
        ``bool(envelope.get("success", False))``, a truthiness test, and
        ``bool("false")``, ``bool("no")``, ``bool("0")`` and
        ``bool(float("nan"))`` are all True. Measured on this tree, the envelope
        ``{"success": "false", "error": "the real error"}`` was adopted and
        re-emitted as::

            {"schema_version": "1.0", "task_type": "t", "success": True,
             "error": "the real error"}

        a FAILED task published as a SUCCESS, carrying its own error beside the
        flag that contradicts it, on the envelope every task consumer keys on.
        The same rule is already written out in
        ``rendering/adapters_validation._coerce_passed``: guessing at truthiness
        "is how an unchecked input renders as a green PASS". A missing ``success``
        key is unchanged (it defaults to the real bool ``False``), and ``0`` /
        ``1`` / ``numpy.bool_`` are still read, so no working caller changes.
        """
        if not isinstance(envelope, dict):
            return cls.fail(
                task_type,
                f"handler returned a non-dict result: {type(envelope).__name__}",
            )
        raw_success = envelope.get("success", False)
        success = _coerce_success(raw_success)
        if success is None:
            reported = envelope.get("error")
            return cls(
                task_type=task_type,
                success=False,
                error=(
                    "handler returned a success flag that is not a boolean: "
                    f"{raw_success!r} ({type(raw_success).__name__}). Adopted as a "
                    "FAILURE, because a truthiness test on it reports SUCCESS for "
                    'the strings "false", "no" and "0" and for NaN.'
                    + (f" The handler's own error was: {reported}" if reported else "")
                ),
                warnings=envelope.get("warnings"),
            )
        return cls(
            task_type=task_type,
            success=success,
            data=envelope.get("data"),
            error=envelope.get("error"),
            warnings=envelope.get("warnings"),
        )

    # Serialization

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the wire envelope.

        Backward compatible with the historical hand-written shape: emits
        ``success`` and, when present, ``data`` / ``error`` / ``warnings``.
        Adds ``schema_version`` and ``task_type`` as additive keys. ``None``
        fields are omitted so a success envelope carries no ``error`` key and
        a failure envelope carries no ``data`` key, exactly as before.
        """
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "success": self.success,
        }
        if self.data is not None:
            out["data"] = self.data
        if self.error is not None:
            out["error"] = self.error
        if self.warnings:
            out["warnings"] = self.warnings
        return out
