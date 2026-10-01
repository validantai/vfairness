"""BGL6 PIN, batch A-operations-4-b (2026-09-29): reporting/compliance.py.

``compute_signed_test_log`` documents three provenance states for the timestamp
it seals, and an independent examiner found the string branch
(``isinstance(declared, str) and declared.strip()``) never PARSED anything, so
the one type the function did not validate was the one type it trusted.
``'banana'`` was sealed with ``test_timestamp_source`` 'declared_by_caller', the
state the code's own table defines as "a time this function could read", and
then folded into content_payload and the 64-char content_hash: a signed
compliance record asserting a provenance state nothing established.

Read with tests/test_bgl6_f06.py, which pinned the datetime and epoch branches of
the same block one day earlier.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import Any

import pytest

from vfairness.operations.reporting.compliance import compute_signed_test_log


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# ===========================================================================
# ROOT CAUSE 5. The signed test log sealed an unparseable string as a time the
# function could read.
# ===========================================================================

_SEALED = ("abc123", "0.1.0")


def _log(declared: Any = "__absent__") -> dict:
    payload: dict = {"metrics": [], "overall_pass": None}
    if declared != "__absent__":
        payload["timestamp"] = declared
    return compute_signed_test_log(payload, *_SEALED)


@pytest.mark.parametrize("declared", ["banana", "2024-13-45", "last tuesday", "2024-01-15T99:99"])
def test_an_unparseable_timestamp_string_is_no_longer_sealed_as_declared(declared):
    """REFUSAL PIN on compute_signed_test_log. The string branch was
    ``isinstance(declared, str) and declared.strip()`` and parsed NOTHING, so
    the one type the function did not validate was the one type it trusted.
    Measured before, with ``compute_signed_test_log({'timestamp': X,
    'metrics': []}, 'abc123', '0.1.0')``::

        X = 'banana'      -> test_timestamp 'banana',     source
                             'declared_by_caller', ZERO warnings
        X = '2024-13-45'  -> test_timestamp '2024-13-45', source
                             'declared_by_caller', ZERO warnings

    while an epoch int on the same run was correctly refused. The block's own
    state table defines 'declared_by_caller' as "the caller declared a time this
    function could read", so a SIGNED compliance record asserted a provenance
    state the code never established, sealed into content_payload and the
    64-char content_hash."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = _log(declared)
    messages = [str(w.message) for w in caught]

    assert log["test_timestamp_source"] == "declared_but_unreadable", log["test_timestamp_source"]
    assert log["test_timestamp"] != declared, "the unreadable value must not be sealed as the time"
    assert log["test_timestamp"] == log["log_built_at"]
    assert any(repr(declared) in m and "cannot" in m for m in messages), messages


def test_control_a_readable_declaration_is_still_sealed_verbatim():
    """OVER-CORRECTION CONTROL. An ISO string is still sealed EXACTLY as the
    caller spelled it (the seal covers the caller's own string, which
    tests/test_bgl4_operations_4.py asserts), a datetime is still normalised
    through .isoformat(), and neither warns. A parser that refused these would
    pass the pin above and break the provenance record it exists to protect."""
    iso, warned = _caught(_log, "2024-01-15T09:00:00+00:00")
    assert iso["test_timestamp"] == "2024-01-15T09:00:00+00:00"
    assert iso["test_timestamp_source"] == "declared_by_caller"
    assert warned == [], warned

    naive, naive_warned = _caught(_log, _dt.datetime(2024, 1, 15, 9, 0, 0))
    assert naive["test_timestamp"] == "2024-01-15T09:00:00"
    assert naive["test_timestamp_source"] == "declared_by_caller"
    assert naive_warned == [], naive_warned

    spaced, spaced_warned = _caught(_log, "2024-01-15 09:00:00")
    assert spaced["test_timestamp"] == "2024-01-15 09:00:00"
    assert spaced["test_timestamp_source"] == "declared_by_caller"
    assert spaced_warned == [], spaced_warned


def test_control_the_other_two_timestamp_states_are_unchanged():
    """OVER-CORRECTION CONTROL. The three states stay three: nothing declared is
    still 'log_build_time' in silence (a blank string included), and an epoch
    number is still 'declared_but_unreadable' with its loud warning. The new
    string parse must not collapse any of them into another."""
    absent, absent_warned = _caught(_log)
    assert absent["test_timestamp_source"] == "log_build_time"
    assert absent["test_timestamp"] == absent["log_built_at"]
    assert absent_warned == [], absent_warned

    blank, blank_warned = _caught(_log, "   ")
    assert blank["test_timestamp_source"] == "log_build_time"
    assert blank_warned == [], blank_warned

    epoch, epoch_warned = _caught(_log, 1705309200)
    assert epoch["test_timestamp_source"] == "declared_but_unreadable"
    assert any("1705309200" in w for w in epoch_warned), epoch_warned


def test_the_seal_covers_the_corrected_provenance_state():
    """The state is sealed, so the correction has to reach the hash. Two logs
    that differ ONLY in whether the declared timestamp could be read must carry
    different content hashes, otherwise the provenance field is decoration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        readable = _log("2024-01-15T09:00:00+00:00")
        unreadable = _log("banana")
    assert len(readable["content_hash"]) == 64
    assert readable["content_hash"] != unreadable["content_hash"]
