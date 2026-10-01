"""The sentence that tells a reader how much to trust an unchecked grade.

Every docstring whose grade has NOT been independently checked carries one
sentence saying so, and that sentence quotes how often such grades have turned
out wrong. It is the single number a reader weighs the grade against.

WHY THIS FILE EXISTS. Until 2026-09-30 that fraction was a hardcoded literal,
"roughly one grade in five", stamped into 104 shipped docstrings. The measured
union rate that day was 291 of 872, which is one grade in THREE, so all 104
understated it by a factor of about 1.7.

Understating it does not fail safe. Every other guard in this repository exists
to stop the library claiming more than it measured, and this one sentence did
exactly that in the reader's favour: it made an unchecked grade sound more
settled than the evidence supports. A stale reassurance is worse than a stale
warning.

The check reads the SHIPPED SOURCE, not the generator, so it holds in the public
repository where the generator is not published, and it would still fail if
somebody pasted the old wording back into a docstring by hand.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import library_kpis  # noqa: E402

SRC = ROOT / "src" / "vfairness"
PHRASE = re.compile(r"Roughly one grade in ([a-z]+|\d+) has been overturned")
_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def _measured_n() -> int:
    g = library_kpis.grading()
    overturned = g.get("audit_overturned_ever") or 0
    audited = g.get("audited_ever") or 0
    assert overturned and audited, (
        "the overturn rate could not be measured, so this test cannot say whether the "
        f"stamped sentence agrees with it: overturned={overturned} audited={audited}"
    )
    return max(2, round(audited / overturned))


def _stamped() -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for raw in PHRASE.findall(path.read_text(encoding="utf-8", errors="replace")):
            n = _WORDS.get(raw, None)
            if n is None:
                n = int(raw) if raw.isdigit() else -1
            # relative_to only when it IS under the repo: the refusal test below
            # points SRC at a temp tree, and a helper that raises on that could not
            # be used to prove this check can fail.
            try:
                shown = str(path.relative_to(ROOT))
            except ValueError:
                shown = str(path)
            found.setdefault(n, []).append(shown)
    return found


def test_the_stamped_fraction_is_the_measured_one():
    expected = _measured_n()
    stamped = _stamped()
    assert stamped, (
        "no docstring carries the sentence at all, so this test is vacuous. Either the "
        "stamper stopped emitting it or every grade is now independently checked; if "
        "the latter, delete this test with the reason."
    )
    wrong = {n: files for n, files in stamped.items() if n != expected}
    assert not wrong, (
        f"the measured overturn rate is one grade in {expected}, and these docstrings "
        f"tell a reader something else: "
        + "; ".join(
            f"one in {n} in {len(files)} file(s), e.g. {files[0]}" for n, files in wrong.items()
        )
        + ". Re-run ./scripts/refresh-docs.sh; the phrase is derived, not written."
    )


def test_the_sentence_appears_only_where_the_grade_is_unchecked():
    """OVER-CORRECTION CONTROL. A stamper that put the warning on every unit would
    pass the test above and tell a reader nothing."""
    stamped_files = {f for files in _stamped().values() for f in files}
    total = sum(1 for _ in SRC.rglob("*.py"))
    assert len(stamped_files) < total, (
        f"every one of the {total} source files carries the unchecked-grade warning, "
        "which would mean it is not distinguishing checked grades from unchecked ones"
    )


@pytest.mark.parametrize("stale", ["five", "ten", "20"])
def test_the_check_would_catch_a_pasted_stale_wording(stale, tmp_path, monkeypatch):
    """The refusal half: prove this test can fail, on a tree that carries the old
    wording, rather than trusting that it would."""
    fake = tmp_path / "vfairness"
    fake.mkdir()
    (fake / "unit.py").write_text(
        f'"""Roughly one grade in {stale} has been overturned when somebody did."""\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "SRC", fake)
    stamped = _stamped()
    assert stamped, "the fixture was not read at all"
    assert all(n != _measured_n() for n in stamped), (
        f"'one grade in {stale}' was read as agreeing with the measurement"
    )
