"""BGL4 audit of batch A-net-1: one grade, overturned, recorded as evidence.

``vfairness.net.egress.GuardedSession.close`` was graded SEMI-PROVEN from probe
evidence alone, with the reason "measured on healthy input and refused on all 8
degenerate worlds that ran". Neither half of that sentence survives execution:

  * ``close`` takes no argument but ``self`` (signature ``(self) -> None``), so
    none of the eight degenerate worlds could vary anything the unit reads. The
    worlds differ in labels, scores and group columns; this unit reads none of
    them.
  * it returns ``None`` on every session state, and ``scripts/surface_probe.py``
    classifies ``None`` as ``("refused", "None")``, so the probe recorded a
    refusal on the HEALTHY world too. ``scripts/grade_from_probe.classify``
    rejects a healthy world only when it is ``raised``, ``hung`` or
    ``NOT_REACHED``, so clause 1 (the healthy world must have produced a value)
    never fired, and "measured on healthy input" is false.

The unit produces no fairness number and no verdict, and it gates none: after
``close`` the guard still refuses a forbidden target, because the cleared cache
makes the next hop re-validate. The corrected grade is NOT A MEASUREMENT, which
is what an agent gave every sibling of this shape in
``docs/surface-grading-2026-09-18.json`` (``CalibrationAnalyzer.clear_cache``,
``BiasDetector.clear_cache``, ``FairnessMonitor.reset``, ``BiasMonitor.reset``).

The two passing tests pin the facts behind that reclassification. The third
recorded a ROBUSTNESS gap found while sweeping the state classes; it is not a
fabrication and it was not the reason for the regrade.

AUDITOR NOTE, SUPERSEDED 2026-09-27 (BGL5): the robustness gap this file recorded
as an ``xfail(strict=True)`` IS now fixed in ``GuardedSession.close``, which closes
every adapter and the parent session before re-raising, so that marker had to go:
under ``xfail_strict`` an XPASS is a failure. The test keeps its subject and now
also checks that ``super().close()`` was reached. The REGRADE itself is not a code
change and is not recorded here; see /tmp/claude-501/bgl/fix/fix-A-net-1.json and
tests/test_bgl5_toplevel_and_net.py.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from vfairness.net.egress import GuardedSession, PinnedIPAdapter, SSRFError

# A cloud-metadata literal, so the guard decides without any DNS lookup.
_METADATA_URL = "http://169.254.169.254/latest/meta-data/"


def _session() -> GuardedSession:
    return GuardedSession(allow_http=True, allow_loopback=True)


def _adapter() -> PinnedIPAdapter:
    return PinnedIPAdapter("localhost", "127.0.0.1", tls=False)


def test_close_reads_no_input_and_reports_no_value(capsys) -> None:
    """The regrade, pinned: nothing measurable in, nothing measurable out.

    Observed 2026-09-27 on every state class below: return None, empty stdout,
    empty stderr, zero warnings, cache emptied.
    """
    assert str(inspect.signature(GuardedSession.close)) == "(self) -> 'None'"
    assert GuardedSession.close.__annotations__ == {"return": "None"}

    states = {
        "empty cache": lambda s: None,
        "one cached adapter": lambda s: s._guard_adapters.update(
            {("http", "localhost", 80): _adapter()}
        ),
        "three cached adapters": lambda s: s._guard_adapters.update(
            {("http", f"h{i}", 80): _adapter() for i in range(3)}
        ),
        "an already closed adapter": lambda s: s._guard_adapters.update(
            {("http", "localhost", 80): _already_closed()}
        ),
    }
    for name, prepare in states.items():
        session = _session()
        prepare(session)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert session.close() is None, name
        assert caught == [], name
        assert session._guard_adapters == {}, name
        # Idempotent: a second close is still no value and still no warning.
        assert session.close() is None, name

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def _already_closed() -> PinnedIPAdapter:
    adapter = _adapter()
    adapter.close()
    return adapter


def test_close_does_not_gate_a_measurement_and_does_not_widen_the_guard() -> None:
    """The other half of the regrade: it gates nothing a reader would act on.

    Clearing the vetted-adapter cache can only make the next hop re-validate, so
    the refusal is still there after close. Observed: SSRFError both times, with
    the same message.
    """
    session = GuardedSession(allow_http=True, allow_loopback=False)
    with pytest.raises(SSRFError) as before:
        session.get_adapter(_METADATA_URL)
    session.close()
    with pytest.raises(SSRFError) as after:
        session.get_adapter(_METADATA_URL)
    assert "non-public address" in str(before.value)
    assert str(after.value) == str(before.value)


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: "ROBUSTNESS, not
# fabrication, and not the reason for the regrade. close() iterates the cache with no
# try/finally, so the first adapter whose close() raises aborts the loop: observed
# 2026-09-27, two adapters left in _guard_adapters and super().close() never reached,
# so the parent Session's own adapters leak too."
def test_close_closes_every_adapter_even_when_one_of_them_raises() -> None:
    """The gap is closed: every adapter and the parent session are closed first and
    the exception still reaches the caller. Measured before the fix, on this exact
    fixture: RuntimeError propagated with 2 adapters left in _guard_adapters and the
    parent session's own adapter never closed."""

    class _Boom(PinnedIPAdapter):
        def close(self) -> None:
            raise RuntimeError("poolmanager gone")

    class _Recorder(PinnedIPAdapter):
        closed = False

        def close(self) -> None:
            type(self).closed = True
            super().close()

    _Recorder.closed = False
    session = _session()
    session._guard_adapters[("http", "boom", 80)] = _Boom("localhost", "127.0.0.1", tls=False)
    session._guard_adapters[("http", "survivor", 80)] = _adapter()
    # The observable for "super().close() was reached".
    session.adapters["sentinel://"] = _Recorder("localhost", "127.0.0.1", tls=False)
    with pytest.raises(RuntimeError):
        session.close()
    assert session._guard_adapters == {}
    assert _Recorder.closed is True, "super().close() was never reached"
