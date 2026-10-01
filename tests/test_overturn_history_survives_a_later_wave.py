"""The published overturn count must not shrink when a later wave re-grades a row.

WHY. The hardening page reports how often the grading method disagreed with
itself, and that figure is the one a reader needs most: it is the only thing on
the page that says whether a grade is worth anything. It was computed from the
MERGED view of the grading waves, where a later wave replaces a row wholesale, so
re-grading a row also deleted the record that its first grade had been overturned.

Measured on 2026-09-27, before the fix: the 2026-09-18 wave recorded 124
overturns out of 410 audited, roughly one in three. Later waves superseded 89 of
those rows. The page therefore published 35, and a reader would have concluded
the method was wrong about one grade in eight. The campaign got MORE thorough and
the number reporting its fallibility fell by a factor of three.

That is this library's own defect committed by its own statistics: a figure that
reads as a measurement of something and is an artifact of a merge.

Two questions, two figures, and neither may be quietly answered with the other:

  ``audit_overturned``       is the grade published TODAY a second opinion?
  ``audit_overturned_ever``  how often has this method ever disagreed with itself?

The first is correctly computed from the merge and correctly falls when a row is
re-graded, because the new grade genuinely has not been audited. The second is a
historical fact and can only grow.

The same reasoning does NOT apply to ``audited``, and this file pins that too. An
audit of a grade a later wave replaced does not transfer to the replacement, so a
superseded row is unaudited again and must be counted that way.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _kpis():
    """Load scripts/library_kpis.py as a module without installing it."""
    path = ROOT / "scripts" / "library_kpis.py"
    spec = importlib.util.spec_from_file_location("_library_kpis_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kpis():
    return _kpis()


def _wave(path: pathlib.Path, measured_on: str, items: dict) -> pathlib.Path:
    path.write_text(
        json.dumps({"measured_on": measured_on, "items": items, "states": {}}),
        encoding="utf-8",
    )
    return path


def _overturned_row() -> dict:
    return {
        "grade": "PROVEN",
        "audited": True,
        "audit_overturned": True,
        "test_file": "tests/test_x.py",
        "sabotage": "reverted the guard, test went red",
    }


def _regraded_row() -> dict:
    return {
        "grade": "PROVEN",
        "test_file": "tests/test_y.py",
        "sabotage": "reverted the guard, test went red",
    }


def _point_at(kpis, tmp_path, base: dict, later: dict | None = None):
    """Aim the module's wave paths at temporary files.

    The mechanism is asked with fixtures rather than with today's evidence,
    because today's evidence happens to contain an overturn and a test that
    passes for that reason alone would go quiet the moment somebody curated it
    away. A check that cannot disagree is the thing this whole suite is about.
    """
    kpis.GRADING = _wave(tmp_path / "surface-grading-2026-01-01.json", "2026-01-01", base)
    kpis.GRADING_FROM_PROBE = _wave(tmp_path / "surface-grading-from-probe.json", "2026-01-01", {})
    kpis.LATER_WAVES = (
        [_wave(tmp_path / "surface-grading-2026-02-02.json", "2026-02-02", later)]
        if later is not None
        else []
    )


def test_an_overturn_is_still_counted_after_a_later_wave_regrades_that_row(kpis, tmp_path):
    """The exact shape that lost 89 overturns: the row is replaced, not amended."""
    _point_at(kpis, tmp_path, base={"a.b.c": _overturned_row()})
    before = kpis.grading()
    assert before["audit_overturned_ever"] == 1, before
    assert before["audit_overturned"] == 1, before

    _point_at(kpis, tmp_path, base={"a.b.c": _overturned_row()}, later={"a.b.c": _regraded_row()})
    after = kpis.grading()

    assert after["audit_overturned_ever"] == 1, (
        "re-grading a row deleted the record that its first grade was overturned; "
        f"ever went to {after['audit_overturned_ever']}"
    )
    assert after["audit_overturned_superseded"] == 1, after


def test_the_current_count_does_fall_when_the_row_is_regraded(kpis, tmp_path):
    """The over-correction control. Both figures rising together would be wrong too.

    A fix that simply made ``audit_overturned`` a union as well would pass the
    test above and publish a false claim: that the grade shown today is a second
    opinion, when the row has been re-graded and nobody has checked the new one.
    """
    _point_at(kpis, tmp_path, base={"a.b.c": _overturned_row()}, later={"a.b.c": _regraded_row()})
    after = kpis.grading()
    assert after["audit_overturned"] == 0, (
        "the row was re-graded and nobody audited the new grade, so today's grade "
        "is not a second opinion and must not be counted as one"
    )
    assert after["audited"] == 0, after
    assert after["unaudited"] == 1, after


def test_an_audit_does_not_transfer_across_a_supersession(kpis, tmp_path):
    """A superseded row is unaudited again, and that is not the same bug."""
    _point_at(kpis, tmp_path, base={"a.b.c": _overturned_row()})
    assert kpis.grading()["audited"] == 1

    _point_at(kpis, tmp_path, base={"a.b.c": _overturned_row()}, later={"a.b.c": _regraded_row()})
    assert kpis.grading()["audited"] == 0


def test_the_history_counts_a_row_once_however_many_waves_flag_it(kpis, tmp_path):
    """It is a union, so a row overturned twice is one row, not two."""
    _point_at(
        kpis,
        tmp_path,
        base={"a.b.c": _overturned_row()},
        later={"a.b.c": _overturned_row()},
    )
    g = kpis.grading()
    assert g["audit_overturned_ever"] == 1, g
    assert g["audit_overturned_superseded"] == 0, g


def test_the_live_history_is_never_below_the_live_current_count():
    """Against the real evidence files, the one relation that must always hold."""
    g = _kpis().grading()
    assert g["available"], g
    assert g["audit_overturned_ever"] >= g["audit_overturned"], g
    assert g["audit_overturned_ever"] - g["audit_overturned"] == g["audit_overturned_superseded"], g


def test_the_published_file_carries_both_figures():
    """A figure the page reaches for and the file does not hold blanks silently."""
    stats = json.loads(
        (ROOT / "docs" / "site" / "data" / "library-stats.json").read_text(encoding="utf-8")
    )
    g = stats["kpis"]["hardening"]["grading"]
    for key in ("audit_overturned", "audit_overturned_ever", "audit_overturned_superseded"):
        assert key in g, f"{key} absent from the published statistics"
        assert isinstance(g[key], int), f"{key} is {g[key]!r}, not a count"
