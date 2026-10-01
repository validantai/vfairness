"""BGL7 PIN, batch A-operations-4-c (2026-09-29): reporting/compliance.py.

The SIBLING of the branch tests/test_bgl6_operations_4_signed_log.py pinned one
day earlier. That fix taught the STRING branch of ``compute_signed_test_log`` to
parse with ``datetime.fromisoformat``; the OBJECT branch went on believing
whatever ``.isoformat()`` returned. ``pd.NaT`` IS an instance of
:class:`datetime.datetime`, and its ``isoformat()`` returns the literal string
``'NaT'``, so the exact provenance falsehood the earlier fix says it ended stayed
reachable, through the most ordinary route there is: the value
``DataFrame.to_dict('records')`` puts in a missing timestamp cell.

MEASURED BEFORE THIS CHANGE (canonical interpreter, OMP_NUM_THREADS=1)::

    compute_signed_test_log({"timestamp": pd.NaT, "metrics": [],
                             "overall_pass": None}, "abc123", "0.1.0")
      -> test_timestamp='NaT'  test_timestamp_source='declared_by_caller'  warnings=0

    rec = pd.DataFrame([{"overall_pass": None, "timestamp": pd.NaT}]).to_dict("records")[0]
    compute_signed_test_log({"timestamp": rec["timestamp"], ...})
      -> test_timestamp='NaT'  test_timestamp_source='declared_by_caller'  warnings=0

    an object whose isoformat() returns "banana"
      -> test_timestamp='banana'  test_timestamp_source='declared_by_caller'  warnings=0

'declared_by_caller' is defined in the function's own state table as "a time this
function could read". Nothing read 'NaT'. The value was then sealed into
``content_payload`` and the 64-char ``content_hash``.

The fix puts ONE reader (``_reads_as_iso_time``) ABOVE both producers of a
candidate string, so a second producer cannot reopen the hole.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from typing import Any

import pandas as pd
import pytest

from vfairness.operations.reporting.compliance import compute_signed_test_log

_SEALED = ("abc123", "0.1.0")


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def _log(declared: Any = "__absent__") -> dict:
    payload: dict = {"metrics": [], "overall_pass": None}
    if not (isinstance(declared, str) and declared == "__absent__"):
        payload["timestamp"] = declared
    return compute_signed_test_log(payload, *_SEALED)


class _IsoformatSaysBanana:
    """The object branch's failure mode with no pandas involved."""

    def isoformat(self) -> str:
        return "banana"


def test_pd_nat_is_no_longer_sealed_as_a_time_the_caller_declared():
    """BEFORE: test_timestamp 'NaT', source 'declared_by_caller', 0 warnings.

    pd.NaT passes ``hasattr(declared, "isoformat")`` and is not a str, so it took
    the object branch, whose only test was that ``.isoformat()`` returned a
    non-blank string. 'NaT' is a non-blank string.
    """
    log, warned = _caught(_log, pd.NaT)

    assert log["test_timestamp_source"] == "declared_but_unreadable", log["test_timestamp_source"]
    assert log["test_timestamp"] != "NaT", "'NaT' must never be sealed as the test time"
    assert log["test_timestamp"] == log["log_built_at"]
    assert len(warned) == 1, warned
    assert "NaT" in warned[0]
    assert "declared_but_unreadable" in warned[0]


def test_the_ordinary_to_dict_records_route_is_refused_too():
    """The reachability half. A missing timestamp cell in a DataFrame becomes
    pd.NaT under ``to_dict('records')``, which is how these records are built, so
    this was not an exotic input."""
    rec = pd.DataFrame([{"overall_pass": None, "timestamp": pd.NaT}]).to_dict("records")[0]
    log, warned = _caught(_log, rec["timestamp"])

    assert log["test_timestamp_source"] == "declared_but_unreadable"
    assert log["test_timestamp"] != "NaT"
    assert len(warned) == 1, warned


def test_an_object_whose_isoformat_does_not_read_as_a_time_is_refused():
    """The general form, with no pandas: ``.isoformat()`` returning A STRING is
    not the same as it returning A TIME."""
    log, warned = _caught(_log, _IsoformatSaysBanana())

    assert log["test_timestamp_source"] == "declared_but_unreadable"
    assert log["test_timestamp"] != "banana"
    assert len(warned) == 1, warned


@pytest.mark.parametrize(
    "declared",
    [pd.NaT, _IsoformatSaysBanana()],
    ids=["pd-NaT", "isoformat-banana"],
)
def test_the_unreadable_object_does_not_seal_the_string_into_the_hash(declared):
    """The seal is the point: whatever the object's isoformat() said must not end
    up anywhere in the hashed document, not just in ``test_timestamp``."""
    log, _warned = _caught(_log, declared)

    assert log["test_timestamp_source"] == "declared_but_unreadable"
    assert len(log["content_hash"]) == 64
    for key, value in log.items():
        if isinstance(value, str):
            assert value not in ("NaT", "banana"), (key, value)


def test_control_a_readable_declaration_is_still_sealed_with_its_real_value():
    """OVER-CORRECTION CONTROL, asserting the healthy case's REAL values.

    A parser applied to the object branch could easily refuse every datetime as
    well, which would pass every assertion above and destroy the provenance
    record this function exists to produce. All four of these must still read as
    'declared_by_caller' with ZERO warnings, and carry the exact string:

    - an ISO string, sealed VERBATIM (the caller's own spelling, which
      tests/test_bgl4_operations_4.py also asserts)
    - a naive datetime, normalised through .isoformat()
    - an aware pd.Timestamp, the readable sibling of the pd.NaT above
    - a bare date
    """
    iso, iso_warned = _caught(_log, "2024-01-15T09:00:00+00:00")
    assert iso["test_timestamp"] == "2024-01-15T09:00:00+00:00"
    assert iso["test_timestamp_source"] == "declared_by_caller"
    assert iso_warned == [], iso_warned

    naive, naive_warned = _caught(_log, _dt.datetime(2024, 1, 15, 9, 0, 0))
    assert naive["test_timestamp"] == "2024-01-15T09:00:00"
    assert naive["test_timestamp_source"] == "declared_by_caller"
    assert naive_warned == [], naive_warned

    stamp, stamp_warned = _caught(_log, pd.Timestamp("2024-01-15T09:00:00+00:00"))
    assert stamp["test_timestamp"] == "2024-01-15T09:00:00+00:00"
    assert stamp["test_timestamp_source"] == "declared_by_caller"
    assert stamp_warned == [], stamp_warned

    day, day_warned = _caught(_log, _dt.date(2024, 1, 15))
    assert day["test_timestamp"] == "2024-01-15"
    assert day["test_timestamp_source"] == "declared_by_caller"
    assert day_warned == [], day_warned


def test_control_the_other_two_states_still_behave():
    """The three states stay three. Nothing declared is still 'log_build_time'
    (NOT the new refusal), and a float nan, the sibling dtype of pd.NaT that has
    no isoformat at all, is still 'declared_but_unreadable'."""
    absent, absent_warned = _caught(_log)
    assert absent["test_timestamp_source"] == "log_build_time"
    assert absent_warned == [], absent_warned

    nan, nan_warned = _caught(_log, float("nan"))
    assert nan["test_timestamp_source"] == "declared_but_unreadable"
    assert len(nan_warned) == 1, nan_warned

    na, na_warned = _caught(_log, pd.NA)
    assert na["test_timestamp_source"] == "declared_but_unreadable"
    assert len(na_warned) == 1, na_warned
