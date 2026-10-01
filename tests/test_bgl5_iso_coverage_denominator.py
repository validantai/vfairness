"""An empty assessable denominator is not 0% ISO 42001 coverage.

BGL5, 2026-09-28, found by the fabricated-verdict sweep rather than by reading.

``generate_iso42001_evidence_map`` computed::

    coverage = round(...) / assessable * 100, 1) if assessable else 0.0

so a map in which NOTHING is assessable reported ``coverage_percent`` 0.0. On a
compliance artefact that reads as "we assessed your controls and none of them is
covered", which is the strongest adverse statement this function can make, out of
an assessment that never happened. The two states are opposite in meaning and
were the same number.

Both halves are pinned here, because only the pair distinguishes the fix from a
function that has stopped answering: an empty wizard STILL earns a real 0.0 over
the 13 controls the map can assess, and only the all-manual map returns None.
"""

import pytest

from vfairness.operations.reporting import compliance as C


@pytest.fixture
def only_manual_controls(monkeypatch):
    """The map narrowed to the two controls it declares it cannot determine.

    A.2.4 and A.6.4 carry ``coverage_status='manual'`` unconditionally: no
    evidence test exists for either, which the source says at both sites. With
    only those two in the list, ``assessable`` is 0 through the public function,
    so this is the real branch and not a hand-built payload.
    """
    manual = [t for t in C._ISO42001_CONTROLS if t[0] in ("A.2.4", "A.6.4")]
    assert len(manual) == 2, f"expected both manual controls, got {manual}"
    monkeypatch.setattr(C, "_ISO42001_CONTROLS", tuple(manual))
    return manual


def test_a_map_with_nothing_assessable_reports_no_percentage(only_manual_controls):
    """Measured 2026-09-28: coverage_percent was 0.0, and is now None."""
    result = C.generate_iso42001_evidence_map({})

    assert result["assessable_controls"] == 0
    assert result["total_controls"] == 2
    assert result["coverage_percent"] is None, (
        f"coverage_percent is {result['coverage_percent']!r} over an empty "
        "assessable denominator, which a reader takes as measured non-compliance"
    )
    basis = result["coverage_basis"]
    assert "NOT ASSESSED" in basis, basis
    assert "not 0% coverage" in basis.lower(), (
        "the basis does not tell the reader that this is not 0% coverage, which is "
        f"the whole distinction: {basis!r}"
    )


def test_control_an_empty_wizard_still_earns_a_real_zero():
    """The over-correction control. A function that answered None to everything
    would pass the pin above, and it would delete a true finding: an empty wizard
    genuinely covers none of the 13 controls this map CAN assess, and 0.0 is the
    measurement, not a substitute for one.
    """
    result = C.generate_iso42001_evidence_map({})

    assert result["assessable_controls"] == 13
    assert result["coverage_percent"] == 0.0, (
        "an empty wizard covers none of the 13 assessable controls, so 0.0 is a "
        f"measured answer; got {result['coverage_percent']!r}"
    )
    assert "NOT ASSESSED" not in result["coverage_basis"]
    assert result["manual"] == 2
