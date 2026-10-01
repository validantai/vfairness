"""The public explanation of grading must state the numbers we actually measured.

A page that explains a method and quotes a figure the evidence does not support
is the defect this library was audited for, committed in prose. These pins are
cheap and they are the reason the 23-of-78 figure in the first draft was caught
before it shipped: the real numbers are 27 of 120.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "GRADING.md"
EVIDENCE = ROOT / "docs" / "surface-grading-2026-09-18.json"


def _items() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))["items"]


def _grading() -> dict:
    """The MERGED view, from the one place it is computed.

    This pin read the 2026-09-18 file alone until 2026-09-27. That was right while
    there was one wave and wrong afterwards: a later wave replaces a row and takes
    its overturn flag with it, so a figure computed from the merged view falls as
    the campaign gets more thorough. Measured that day, the single-file reading was
    123 of 409 and the merged one was 34 of 277. This asks library_kpis for the
    union, which is the same number the site and the generated doc blocks publish,
    so the three cannot disagree.
    """
    spec = importlib.util.spec_from_file_location(
        "_kpis_for_grading_doc", ROOT / "scripts" / "library_kpis.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.grading()


def test_the_overturn_figure_is_the_measured_one() -> None:
    g = _grading()
    text = DOC.read_text(encoding="utf-8")
    expected = f"{g['audit_overturned_ever']} of {g['audited_ever']}"
    assert expected in text, (
        f"GRADING.md does not state the measured overturn rate ({expected}). It is a "
        "union over every wave file, numerator AND denominator, because a row a "
        "later wave re-graded still went through the independent check once."
    )


def test_the_overturn_denominator_is_not_the_merged_one() -> None:
    """Over-correction control: the smaller, merged figure must NOT be what is stated.

    Without this, swapping the union back for the merged view would pass the test
    above by simply restating whatever the code computed, which is how the figure
    silently fell by a factor of three in the first place.
    """
    g = _grading()
    text = DOC.read_text(encoding="utf-8")
    merged = f"{g['audit_overturned']} of {g['audited']}"
    if merged == f"{g['audit_overturned_ever']} of {g['audited_ever']}":
        return  # nothing has been superseded yet; the two readings coincide
    assert merged not in text, (
        f"GRADING.md states {merged}, the MERGED overturn count. That undercounts how "
        "often this method disagreed with itself, because re-grading a row deletes "
        "the record that its first grade was wrong."
    )


def test_the_not_a_measurement_figure_is_the_measured_one() -> None:
    items = _items()
    claimed = [v for v in items.values() if v.get("claimed_grade") == "NOT A MEASUREMENT"]
    wrong = [v for v in claimed if v.get("audit_overturned")]
    text = DOC.read_text(encoding="utf-8")
    assert f"{len(wrong)} of {len(claimed)}" in text, (
        f"GRADING.md does not state the measured figure ({len(wrong)} of {len(claimed)})"
    )


def test_the_explanation_states_all_five_states() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    for state in ("proven", "semi-proven", "unproven", "defect open", "not a measurement"):
        assert state in text, f"GRADING.md does not explain the state {state!r}"


def test_it_says_a_grade_is_not_a_guarantee() -> None:
    """The limit matters as much as the claim. A page that says what a grade
    proves and not what it fails to prove is the two-state failure in prose."""
    text = DOC.read_text(encoding="utf-8")
    assert "does **not** mean" in text or "does not mean" in text
    assert "statistics are correct" in text


def test_it_does_not_add_the_two_measurements_together() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "never added together" in text or "Adding 215 and 420" in text
