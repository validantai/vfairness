"""The row-level register's claim about its own pins must be checkable.

``docs/audits/row-level-fabrication-register-2026-08-27.md`` closed 22 numbered
entries (R-01 to R-22) and asserted, in the state table at the top of the file:

    Each is pinned by a named passing test in the six ``tests/test_rowlevel_*.py``
    files.

That sentence named no test. A reader could not check it by lookup, and it was
also FALSE: ``grep -rc "R-01" tests/test_rowlevel_*.py`` (and the same for the
other 21) returns 0, and two entries are not pinned in those six files at all.
R-19 is pinned in ``tests/test_explain_partial_runs.py``, and R-20's ten
``explain.py`` finders are pinned by a parametrised test in the same file; nine
of those ten finder names appear nowhere in the six files.

That is the register's own defect class turned on the register: a statement that
reads as assessed while nothing assessed it. So each row now ends with the pytest
node id that pins it, and this file executes those ids.

WHAT THIS ESTABLISHES. Every R-NN row names a pin; each named id resolves to at
least one collected test; and running all of them reports nothing but passes, so
no id is quietly SKIPPED or ``xfail``-marked. A green ``xfail`` is a note, not a
pin - that is exactly N-22, which sat in this campaign's own suite advertising
three defects that no longer existed while being unable to fail in either
direction.

WHAT IT DOES NOT ESTABLISH. It does not judge whether a named test asserts the
right thing. Only a person reading the test against the entry can do that; the
wave records and the reproductions in the register are where that lives.

The checker has its own positive controls in ``TestTheCheckerItselfIsChecked``:
a node id that does not exist, one that is skipped, and one that is ``xfail``
must each be reported, and a healthy pair must not be. Without them a checker
that silently found nothing to run would pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
AUDITS = LIB_ROOT / "docs" / "audits"
REGISTER = AUDITS / "row-level-fabrication-register-2026-08-27.md"

# `docs/audits` is on the export script's EXCLUDES list, so the public repo has
# no audits directory at all. Skip there rather than failing on day one, which
# is a defect this project has already shipped once (a test that read a path
# existing only in the private monorepo turned 39 tests red on the first push).
# The skip is narrow on purpose: if the directory EXISTS, the register must, so
# a rename cannot turn this file into a silent no-op in the repo that has it.
if not AUDITS.is_dir():
    pytest.skip(
        "docs/audits is not part of this checkout (it is excluded from the public export)",
        allow_module_level=True,
    )

# `| R-04 | ... | ... Pinned by `tests/x.py::Test::test_y`. |`
_ROW_RE = re.compile(r"^\|\s*(R-\d\d)\s*\|.*$", re.MULTILINE)
_PIN_RE = re.compile(r"Pinned by `([^`]+)`")

EXPECTED_IDS = tuple(f"R-{n:02d}" for n in range(1, 23))


def register_pins(text: str) -> Dict[str, str]:
    """Map each ``R-NN`` row to the node id its State cell names.

    Rows are read from the markdown rather than hard-coded here, so editing the
    register is what changes what this file checks. A row with no ``Pinned by``
    is reported by :func:`missing_pins` rather than dropped silently.
    """
    pins: Dict[str, str] = {}
    for match in _ROW_RE.finditer(text):
        row = match.group(0)
        rid = match.group(1)
        found = _PIN_RE.findall(row)
        if len(found) == 1:
            pins[rid] = found[0]
        elif len(found) > 1:
            pins[rid] = "\x00multiple\x00" + " and ".join(found)
    return pins


def missing_pins(text: str) -> List[str]:
    """Rows that exist but name no single pin, and expected rows that are absent."""
    pins = register_pins(text)
    rows = {m.group(1) for m in _ROW_RE.finditer(text)}
    problems: List[str] = []
    for rid in EXPECTED_IDS:
        if rid not in rows:
            problems.append(f"{rid}: the register has no row for it any more")
        elif rid not in pins:
            problems.append(f"{rid}: the row names no pin ('Pinned by `<node id>`' is missing)")
        elif pins[rid].startswith("\x00multiple\x00"):
            problems.append(f"{rid}: the row names several pins: {pins[rid].split(chr(0))[2]}")
    return problems


# --- Running the named ids ---------------------------------------------------

# pytest prints these with the node id when a test does not simply pass.
_OUTCOME_RE = re.compile(r"^(FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE)
_NOT_FOUND_RE = re.compile(r"^ERROR: not found: (\S+)", re.MULTILINE)


def _covers(pin: str, node_id: str) -> bool:
    """Does ``node_id`` belong to ``pin``? Exact, or one case of a parametrised pin."""
    return node_id == pin or node_id.startswith(pin + "[")


def unmet_pins(pins: Sequence[str], root: Path = LIB_ROOT) -> List[str]:
    """Execute the named node ids and report every one that is not a passing test.

    One subprocess for all of them: nested in-process pytest inherits this run's
    config and plugins, and a per-id subprocess would cost 22 interpreter starts
    to say the same thing. ``-p no:randomly`` keeps the ids stable, ``--no-cov``
    keeps the parent run's coverage plugin out of the child.
    """
    if not pins:
        return ["no pins were supplied, so nothing was executed"]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *pins,
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-rfEsxX",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    problems: List[str] = []

    for node_id in _NOT_FOUND_RE.findall(output):
        problems.append(
            f"{node_id}: no such test. The register names a pin that pytest cannot collect, "
            "so the claim that the entry is pinned cannot be checked."
        )

    for outcome, node_id in _OUTCOME_RE.findall(output):
        owner = next((p for p in pins if _covers(pin=p, node_id=node_id)), node_id)
        if outcome in {"SKIPPED", "XFAIL", "XPASS"}:
            problems.append(
                f"{owner}: reported {outcome} for {node_id}. A pin that does not run, or that "
                "is marked xfail, records a defect without being able to fail if it returns."
            )
        else:
            problems.append(f"{owner}: reported {outcome} for {node_id}.")

    if completed.returncode != 0 and not problems:
        problems.append(
            f"pytest exited {completed.returncode} running the named pins, with no per-test "
            f"outcome to attribute it to:\n{output[-1500:]}"
        )
    return problems


# --- The pins ----------------------------------------------------------------


def test_the_register_still_carries_the_twenty_two_rows_this_file_checks() -> None:
    """Guard against a vacuous run: the parser must find something to check."""
    text = REGISTER.read_text(encoding="utf-8")
    pins = register_pins(text)
    assert not missing_pins(text), "\n".join(missing_pins(text))
    assert len(pins) == 22, f"expected 22 pinned rows, parsed {len(pins)}: {sorted(pins)}"


def test_every_named_pin_resolves_to_a_passing_test() -> None:
    """The register's claim, executed rather than read.

    Slow by the standards of this suite (it starts one child pytest), and that
    is the point: the previous version of the claim cost nothing and proved
    nothing.
    """
    text = REGISTER.read_text(encoding="utf-8")
    pins = register_pins(text)
    problems = unmet_pins(sorted(set(pins.values())))
    assert not problems, (
        "docs/audits/row-level-fabrication-register-2026-08-27.md names pins that are not "
        "passing tests:\n\n" + "\n".join(problems)
    )


def test_the_register_does_not_reinstate_the_unfalsifiable_claim() -> None:
    """The sentence that started this: a blanket claim naming no test.

    It read as though it had been checked, was unverifiable by lookup, and was
    wrong about two of the 22. If it comes back, the per-row pins have probably
    been dropped with it.
    """
    text = REGISTER.read_text(encoding="utf-8")
    banned = "Each is pinned by a named passing test in the six"
    body = text.split("### Correction, second direction", 1)[0]
    assert banned not in body, (
        "the register asserts again that every entry is pinned by a test in the six "
        "tests/test_rowlevel_*.py files, without naming one. Two of them are not pinned "
        "there at all."
    )


# --- The checker, checked ----------------------------------------------------


_HEALTHY_MODULE = """
def test_one_that_passes():
    assert True


def test_another_that_passes():
    assert True
"""

_UNHEALTHY_MODULE = """
import pytest


@pytest.mark.skip(reason="planted")
def test_one_that_is_skipped():
    assert True


@pytest.mark.xfail(reason="planted", strict=False)
def test_one_that_is_xfail():
    assert True
"""


class TestTheCheckerItselfIsChecked:
    """Plant each way a pin can be hollow, and require the checker to report it.

    A checker that has only ever seen a healthy register is not known to be a
    checker. Everything here runs against files written into ``tmp_path``; no
    repository file is touched, and no pin in the real register is relied on,
    so a wave that deletes an xfail marker elsewhere cannot silently disarm
    these controls.
    """

    def _tree(self, tmp_path: Path, name: str, body: str) -> Path:
        (tmp_path / name).write_text(body, encoding="utf-8")
        return tmp_path

    def test_a_healthy_pair_is_not_flagged(self, tmp_path: Path) -> None:
        root = self._tree(tmp_path, "test_healthy_probe.py", _HEALTHY_MODULE)
        problems = unmet_pins(
            [
                "test_healthy_probe.py::test_one_that_passes",
                "test_healthy_probe.py::test_another_that_passes",
            ],
            root=root,
        )
        assert problems == [], problems

    def test_a_node_id_that_does_not_exist_is_reported(self, tmp_path: Path) -> None:
        root = self._tree(tmp_path, "test_healthy_probe.py", _HEALTHY_MODULE)
        problems = unmet_pins(
            [
                "test_healthy_probe.py::test_one_that_passes",
                "test_healthy_probe.py::test_a_name_nobody_wrote",
            ],
            root=root,
        )
        assert problems, "a pin naming a test that does not exist passed the checker"
        assert any("test_a_name_nobody_wrote" in p for p in problems), problems

    def test_a_skipped_pin_is_reported(self, tmp_path: Path) -> None:
        root = self._tree(tmp_path, "test_unhealthy_probe.py", _UNHEALTHY_MODULE)
        problems = unmet_pins(["test_unhealthy_probe.py::test_one_that_is_skipped"], root=root)
        assert problems, "a pin that never runs passed the checker"
        assert any("SKIPPED" in p for p in problems), problems

    def test_an_xfail_marked_pin_is_reported(self, tmp_path: Path) -> None:
        """N-22 in miniature: the marker outlived its defect and could not fail."""
        root = self._tree(tmp_path, "test_unhealthy_probe.py", _UNHEALTHY_MODULE)
        problems = unmet_pins(["test_unhealthy_probe.py::test_one_that_is_xfail"], root=root)
        assert problems, "an xfail-marked pin passed the checker"
        assert any("XPASS" in p or "XFAIL" in p for p in problems), problems

    def test_a_row_with_no_pin_is_reported(self) -> None:
        text = "| R-01 | site | defect | CLOSED wave 13. |\n"
        problems = missing_pins(text)
        assert any("R-01" in p and "names no pin" in p for p in problems), problems

    def test_a_missing_row_is_reported(self) -> None:
        problems = missing_pins("| R-01 | s | d | CLOSED. Pinned by `t.py::test_a`. |\n")
        assert any("R-22" in p and "no row" in p for p in problems), problems

    def test_a_row_naming_two_pins_is_reported(self) -> None:
        text = "| R-01 | s | d | Pinned by `a.py::test_a`. Pinned by `b.py::test_b`. |\n"
        problems = missing_pins(text)
        assert any("R-01" in p and "several pins" in p for p in problems), problems

    def test_a_parametrised_pin_is_matched_by_its_cases(self) -> None:
        # R-20's pin is parametrised, so the checker must attribute
        # `...::test_x[case]` to the pin `...::test_x` rather than treating the
        # bare id as uncollected.
        assert _covers("a.py::test_x", "a.py::test_x")
        assert _covers("a.py::test_x", "a.py::test_x[one: a case]")
        assert not _covers("a.py::test_x", "a.py::test_x_longer")
