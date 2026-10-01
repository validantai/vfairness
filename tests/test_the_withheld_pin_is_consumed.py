"""A withheld pin is evidence, so something has to read it.

A grading row whose DECIDING evidence is an internal harness cannot name that
harness in ``test_file``: that string propagates into the published ledger and
the public export deletes the harness, so a reader follows a dead link. Such
rows name it in ``test_file_withheld_from_export`` instead.

WHY THIS FILE EXISTS. On 2026-09-30 that field was read by NOTHING. No script,
no generator, no document and no test consumed it. Two independent examiners in
a row then read only ``test_file`` and both reported the row as naming no
evidence that executes the unit, which is the finding the field exists to
answer. A correct record that no consumer reads is the exact defect this
repository is audited for, committed in the evidence register itself.

The pin is written against SYNTHETIC records on purpose. Every wave file that
actually carries the field is excluded from the public export, so a pin reading
those files would find zero rows in the public repository and pass vacuously.
A test that cannot fail is what this whole campaign is about.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_grade_attribution.py"


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location("_vga_pin", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: A synthetic harness name. Deliberately not the real one: naming a path the
#: export drops would itself publish a dead link out of this docstring, which is
#: the sibling of the defect being pinned.
HARNESS = "tests/test_a_withheld_internal_harness_pin.py"


def test_the_withheld_pin_is_counted_as_evidence(verifier):
    named = verifier._named_files(
        {
            "test_file": "tests/test_public_one.py",
            "test_file_withheld_from_export": f"{HARNESS} is the pin that discriminates",
        }
    )
    assert HARNESS in named, (
        "the withheld pin is the evidence that usually DISCRIMINATES for its row, "
        f"and the verifier did not see it: {named}"
    )
    assert "tests/test_public_one.py" in named, named


def test_a_row_declaring_no_direct_test_still_surrenders_its_withheld_pin(verifier):
    """The ``none`` short circuit must read test_file, not the joined string."""
    named = verifier._named_files(
        {
            "test_file": "none needed for the body",
            "test_file_withheld_from_export": f"{HARNESS} is the pin",
        }
    )
    assert named == [HARNESS], named


def test_a_row_declaring_no_direct_test_and_no_withheld_pin_is_still_exempt(verifier):
    """The over-correction control: the ``none`` exemption must survive."""
    assert (
        verifier._named_files({"test_file": "none: already pinned by tests/test_elsewhere.py"})
        == []
    )


@pytest.mark.parametrize(
    "withheld",
    [None, "", "   "],
    ids=["present_holding_None", "empty", "whitespace"],
)
def test_an_absent_withheld_pin_changes_nothing(verifier, withheld):
    """``.get(key, default)`` does NOT fire when the key is PRESENT holding None."""
    record = {"test_file": "tests/test_public_one.py", "test_file_withheld_from_export": withheld}
    assert verifier._named_files(record) == ["tests/test_public_one.py"]
