"""The published package named the private platform's internal layout, and the gate passed it.

Fourteen package modules the export publishes, three of the published documents
and one test module cited files that exist only in the consuming platform's
private repository: frontend source paths, a planning note, two schema-migration
filenames, an internal package-script command, and the deployment host by
nickname. Every one of them was prose inside a comment or a docstring, written
by someone who could open the file they were citing. The reasoning was worth
keeping and has been kept; the citation was recon that a reader of a public
library cannot act on, because the file is not there for them to open.

THIS IS THE DEFECT CLASS, NOT A TYPO SWEEP. The publish boundary reported
``Leak gate PASSED (denylist + gitleaks, both required)`` over a tree holding
all of it. Nothing was measured and refused; there was simply no pattern for
this class, and the absence of a check was read as a clean verdict. Measured
2026-09-10 at f7f10f34a on the tree the emit-tree diagnostic writes: the gate
exited 0, gitleaks reported "no leaks found" over 23.04 MB of 805 files, and 30
lines across 18 of those files named the private platform.

IT REACHED THE ARTIFACTS, NOT ONLY THE REPOSITORY. Built from that same tree,
the sdist and the wheel each carried 21 of those lines, in 14 modules. The three
documents and the test module are not packaged, so their share stops at the
public repository, and that is the only part of this that GitHub alone would
have contained. The rest is the step that turns this from a hosting question
into a PyPI one: an sdist is published fully readable beside the wheel, and once
uploaded neither can be edited.

THREE STATES. A pattern that cannot be evaluated is could-not-check and must
never read as a clean pass, which is the failure this module's second half is
built to catch: the export script has already shipped a denylist entry that
could not match anything (a pattern beginning with dashes, which grep parsed as
options and exited 2 on), and it looked exactly like a pattern that found
nothing. Files that cannot be decoded are collected separately and asserted
empty rather than skipped quietly, and every scan carries measured floors so
that a scan which read nothing cannot report a clean tree.

NOTHING FORBIDDEN IS WRITTEN DOWN HERE, and that is a design constraint rather
than luck. This module ships. It hard-codes no internal token and no probe: the
probes are SYNTHESISED at run time from the live denylist in the export script,
which is not part of the public repository, so there is nothing in this source
to grep for and nothing in it to reassemble.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"

# Same reasoning as the sibling publish-boundary modules: the export script
# exists only in the private monorepo, so in the public repository there is no
# gate to run and no denylist to derive a probe from. Skipping is the honest
# answer there; failing would train the next reader to ignore a red
# publish-boundary test.
pytestmark = pytest.mark.skipif(
    not EXPORT_SCRIPT.is_file(),
    reason=(
        "scripts/export-vfairness-to-public.sh is not present. It exists only in "
        "the private monorepo, which is the only place an export can run from."
    ),
)

_HAS_GITLEAKS = shutil.which("gitleaks") is not None
needs_gitleaks = pytest.mark.skipif(
    not _HAS_GITLEAKS,
    reason=(
        "gitleaks is not installed. The export gate requires it and refuses "
        "without it, so a refusal could not be told apart from that refusal and "
        "a pass could not happen at all."
    ),
)

#: Suffixes the gate's ``grep -rI`` reads as text, so they are the surface this
#: module has to scan. Kept in step with the sibling export modules.
_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".cff",
        ".html",
        ".ipynb",
        ".js",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

#: Suffixes a scan is allowed to leave unread. Anything else that fails to
#: decode is could-not-check and gets reported, never skipped quietly.
_MEDIA_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".ttf", ".svg"}
)

#: Measured floor on the recon denylist, taken 2026-09-10 with 26 entries live.
#: Deleting one is a deliberate act; this is the tripwire that makes it visible,
#: because a deleted pattern leaves every other assertion in this file green.
_RECON_FLOOR = 26


# ---------------------------------------------------------------------------
# Reading the live lists out of the export script
# ---------------------------------------------------------------------------
# Parsed rather than restated, for the reason the sibling modules give: a copy
# of the list would go on being checked long after the script stopped using it.
# The parser is deliberately duplicated instead of imported, so this module
# keeps working whatever happens to its siblings.


@functools.lru_cache(maxsize=None)
def _script_array(name: str) -> tuple[str, ...]:
    """The entries of a one-per-line bash array in the export script."""
    src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    opened = re.search(rf"^{name}=\(", src, re.M)
    assert opened, f"could not find the {name} array in {EXPORT_SCRIPT}"
    rest = src[opened.end() :]
    head, _, _tail = rest.partition("\n")
    if ")" in head:
        body = head.split(")", 1)[0]
    else:
        closed = re.search(r"^\)", rest, re.M)
        assert closed, f"the {name} array is never closed"
        body = rest[: closed.start()]
    body = re.sub(r"#[^\n]*", "", body)
    entries = re.findall(r"""["']([^"']+)["']""", body)
    assert entries, f"parsed no entries out of {name}"
    return tuple(entries)


@functools.lru_cache(maxsize=None)
def _recon_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(source, compiled) for the recon half, with the script's own casing rule.

    Compilation failures are surfaced as an assertion rather than swallowed: a
    pattern this module cannot compile is one it cannot check, and that is
    could-not-check.
    """
    out: list[tuple[str, re.Pattern[str]]] = []
    broken: list[str] = []
    for pat in _script_array("LEAK_PATTERNS_RECON"):
        try:
            out.append((pat, re.compile(pat, re.IGNORECASE)))
        except re.error as exc:  # pragma: no cover - a broken denylist is a red build
            broken.append(f"{pat}: {exc}")
    assert not broken, (
        "these recon denylist entries do not compile, so nothing in this module "
        "checked them and nothing here may report a pass:\n  " + "\n  ".join(broken)
    )
    return tuple(out)


def _is_dropped(rel: str) -> bool:
    """Would the export delete this subtree-relative path?"""
    for entry in _script_array("EXCLUDES"):
        trimmed = entry.rstrip("/")
        if rel == trimmed or rel.startswith(trimmed + "/"):
            return True
    name = PurePosixPath(rel).name
    for glob in _script_array("EXCLUDE_GLOBS"):
        if PurePosixPath(name).match(glob):
            return True
    return False


@functools.lru_cache(maxsize=None)
def _published_text_files() -> tuple[str, ...]:
    """Every tracked, non-excluded, text-shaped path the export would publish.

    Read from the working tree, not from a commit: a fix that has not been
    committed yet still has to satisfy the boundary, and a scan that only ever
    looked at HEAD would report the state of the last commit while calling it
    the state of the export.
    """
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", LIB_ROOT.name],
        capture_output=True,
        check=False,
    )
    assert listing.returncode == 0, (
        "git ls-files failed, so the published file set could not be enumerated. "
        f"That is could-not-check: {listing.stderr.decode(errors='replace').strip()}"
    )
    prefix = LIB_ROOT.name + "/"
    out: list[str] = []
    for raw in listing.stdout.decode().split("\0"):
        if not raw.startswith(prefix):
            continue
        rel = raw[len(prefix) :]
        if _is_dropped(rel):
            continue
        if PurePosixPath(rel).suffix not in _TEXT_SUFFIXES:
            continue
        if (LIB_ROOT / rel).is_file():
            out.append(rel)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


@dataclass
class _Scan:
    """What the sweep found, and what it could not look at."""

    findings: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    files_read: int = 0
    bytes_read: int = 0
    patterns: int = 0


@functools.lru_cache(maxsize=None)
def _scan_published_tree() -> _Scan:
    scan = _Scan(patterns=len(_recon_patterns()))
    for rel in _published_text_files():
        path = LIB_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            if PurePosixPath(rel).suffix not in _MEDIA_SUFFIXES:
                scan.unreadable.append(f"{rel}: {exc}")
            continue
        scan.files_read += 1
        scan.bytes_read += len(text)
        for number, line in enumerate(text.splitlines(), start=1):
            for pat, rx in _recon_patterns():
                found = rx.search(line)
                if found:
                    scan.findings.append(f"{rel}:{number}: {found.group(0)!r} matches {pat}")
    return scan


# ---------------------------------------------------------------------------
# Synthesising a matching sample for a live pattern, never writing one down
# ---------------------------------------------------------------------------
# The samples below are the only way to prove by EXECUTION that the real gate
# still refuses each class, and they may not be pasted in: a value the gate
# matches is by construction a value that would abort the export on this file.
# So they are built from the live denylist at run time by a small sampler over
# the regex subset the denylist uses. It also means a pattern added tomorrow is
# covered the day it lands, with no edit here.


class _UnsamplableError(Exception):
    """The sampler met a construct it will not guess a value for."""


def _sample(pattern: str) -> str:
    """A short string the given denylist regex matches.

    Handles the constructs the denylist actually uses: literals, escaped
    characters, character classes with ranges, groups, alternation, ``+``,
    ``?``, ``*`` and ``{n}``. Anything else raises, so an unsampled pattern is
    reported as could-not-check instead of quietly dropping out of the control.
    """
    out, index = _sample_alternation(pattern, 0, len(pattern))
    assert index == len(pattern), f"sampler stopped at {index} of {len(pattern)} in {pattern!r}"
    return out


def _sample_alternation(pattern: str, start: int, stop: int) -> tuple[str, int]:
    """Sample the FIRST branch of an alternation, which is enough to match."""
    out, index = _sample_sequence(pattern, start, stop)
    while index < stop and pattern[index] == "|":
        # Skip the remaining branches: one matching branch is a match.
        depth = 0
        index += 1
        while index < stop:
            char = pattern[index]
            if char == "\\":
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            index += 1
    return out, index


def _sample_sequence(pattern: str, start: int, stop: int) -> tuple[str, int]:
    out: list[str] = []
    index = start
    while index < stop:
        char = pattern[index]
        if char in {"|", ")"}:
            break
        piece, index = _sample_atom(pattern, index, stop)
        piece, index = _apply_quantifier(pattern, index, stop, piece)
        out.append(piece)
    return "".join(out), index


def _sample_atom(pattern: str, index: int, stop: int) -> tuple[str, int]:
    char = pattern[index]
    if char == "(":
        inner_start = index + 1
        if pattern.startswith("?:", inner_start):
            inner_start += 2
        depth = 0
        cursor = inner_start
        while cursor < stop:
            here = pattern[cursor]
            if here == "\\":
                cursor += 2
                continue
            if here == "(":
                depth += 1
            elif here == ")":
                if depth == 0:
                    break
                depth -= 1
            cursor += 1
        if cursor >= stop:
            raise _UnsamplableError(f"unclosed group in {pattern!r}")
        inner, _ = _sample_alternation(pattern, inner_start, cursor)
        return inner, cursor + 1
    if char == "[":
        cursor = index + 1
        if cursor < stop and pattern[cursor] == "^":
            raise _UnsamplableError(f"negated class in {pattern!r}")
        body_start = cursor
        while cursor < stop and pattern[cursor] != "]":
            if pattern[cursor] == "\\":
                cursor += 1
            cursor += 1
        if cursor >= stop:
            raise _UnsamplableError(f"unclosed class in {pattern!r}")
        return _first_member(pattern[body_start:cursor]), cursor + 1
    if char == "\\":
        if index + 1 >= stop:
            raise _UnsamplableError(f"trailing escape in {pattern!r}")
        escaped = pattern[index + 1]
        if escaped in {"b", "B", "A", "Z"}:
            return "", index + 2  # zero width
        if escaped == "d":
            return "7", index + 2
        if escaped in {"w", "S"}:
            return "x", index + 2
        if escaped == "s":
            return " ", index + 2
        return escaped, index + 2
    if char in {"^", "$"}:
        return "", index + 1
    if char == ".":
        return "x", index + 1
    if char in {"*", "+", "?", "{"}:
        raise _UnsamplableError(f"quantifier without an atom in {pattern!r}")
    return char, index + 1


def _apply_quantifier(pattern: str, index: int, stop: int, piece: str) -> tuple[str, int]:
    if index >= stop:
        return piece, index
    char = pattern[index]
    if char in {"+", "*"}:
        index += 1
        if index < stop and pattern[index] == "?":
            index += 1
        # One repetition satisfies "+"; for "*" one is also a match.
        return piece, index
    if char == "?":
        index += 1
        return piece, index
    if char == "{":
        closing = pattern.find("}", index)
        if closing == -1:
            raise _UnsamplableError(f"unclosed repetition in {pattern!r}")
        spec = pattern[index + 1 : closing]
        low = spec.split(",")[0]
        if not low.isdigit():
            raise _UnsamplableError(f"unsupported repetition {spec!r} in {pattern!r}")
        return piece * int(low), closing + 1
    return piece, index


def _first_member(body: str) -> str:
    """One character a class body admits, avoiding characters that end a word."""
    cursor = 0
    members: list[str] = []
    while cursor < len(body):
        if body[cursor] == "\\":
            members.append(body[cursor + 1] if cursor + 1 < len(body) else "x")
            cursor += 2
            continue
        if cursor + 2 < len(body) and body[cursor + 1] == "-":
            members.append(body[cursor])
            cursor += 3
            continue
        members.append(body[cursor])
        cursor += 1
    for candidate in members:
        if candidate.isalnum():
            return candidate
    assert members, f"empty character class {body!r}"
    return members[0]


@dataclass
class _Samples:
    """One synthesised sample per live recon pattern, plus what could not be."""

    lines: dict[str, str] = field(default_factory=dict)
    unsamplable: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)


@functools.lru_cache(maxsize=None)
def _recon_samples() -> _Samples:
    samples = _Samples()
    for pat, rx in _recon_patterns():
        try:
            candidate = _sample(pat)
        except (_UnsamplableError, AssertionError) as exc:
            samples.unsamplable.append(f"{pat}: {exc}")
            continue
        if not rx.search(candidate):
            samples.mismatched.append(f"{pat}: synthesised {candidate!r}, which it does not match")
            continue
        samples.lines[pat] = candidate
    return samples


def _run_gate(directory: Path) -> subprocess.CompletedProcess[str]:
    """Execute the REAL leak gate over a directory, read only."""
    return subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--gate-only", str(directory)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def _benign_tree(destination: Path) -> int:
    """Copy a handful of the library's own published sources into a directory.

    The control tree has to be made of real published text, not of invented
    filler: filler proves the gate is quiet about nothing in particular, while
    this proves it is quiet about the very files the export ships.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in _published_text_files():
        if PurePosixPath(rel).suffix != ".py":
            continue
        source = LIB_ROOT / rel
        if source.stat().st_size > 40_000:
            continue
        target = destination / rel.replace("/", "__")
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
        if copied >= 40:
            break
    return copied


# ---------------------------------------------------------------------------
# 1. The refusal pin: nothing the export publishes may name the private platform
# ---------------------------------------------------------------------------


def test_no_published_file_names_the_private_platform() -> None:
    """REFUSAL PIN for the class, over the surface that actually ships.

    Runs the LIVE recon denylist over every file the export would publish,
    read from the working tree. It fails on the 30 sites this change fixed and
    on any future one, whatever shape it arrives in, because the list it uses
    is the gate's own.
    """
    scan = _scan_published_tree()
    assert not scan.findings, (
        "these files the export publishes name internal infrastructure the "
        "published library cannot refer to:\n  " + "\n  ".join(scan.findings)
    )


def test_the_sweep_actually_read_the_published_tree() -> None:
    """The control that makes the pin above mean something.

    A sweep that enumerated nothing, read nothing, or loaded an empty denylist
    reports exactly the same clean result as a clean tree. Each floor below is a
    measured value from 2026-09-10, held well under the real figure so ordinary
    growth does not move it.
    """
    scan = _scan_published_tree()
    assert scan.patterns >= _RECON_FLOOR, (
        f"only {scan.patterns} recon patterns were loaded from the export "
        f"script; the sweep needs at least {_RECON_FLOOR} to be checking the "
        "class this module exists for."
    )
    assert scan.files_read >= 300, (
        f"the sweep read only {scan.files_read} published files. It is not "
        "looking at the export surface, so its clean result is about nothing."
    )
    assert scan.bytes_read >= 1_000_000, (
        f"the sweep read only {scan.bytes_read} characters, which is far below "
        "the published text of this library."
    )
    assert not scan.unreadable, (
        "these published files could not be decoded, so nothing checked them. "
        "That is could-not-check, not a pass:\n  " + "\n  ".join(scan.unreadable)
    )


# ---------------------------------------------------------------------------
# 2. The over-correction control: the gate can still REFUSE, and still ACCEPT
# ---------------------------------------------------------------------------


@needs_gitleaks
def test_the_real_gate_refuses_every_class_in_its_own_denylist(tmp_path: Path) -> None:
    """OVER-CORRECTION CONTROL, asserting MEASURED refusals rather than absence.

    The pin above goes green both when the tree is clean and when the denylist
    has quietly stopped being able to match anything. This half executes the
    real gate over a tree built from one synthesised sample per live pattern,
    and asserts that it refuses AND that its report names every one of them.
    A pattern that runs but can never match is the failure this catches; the
    script has shipped one before.
    """
    samples = _recon_samples()
    assert not samples.unsamplable, (
        "no sample could be built for these recon patterns, so nothing proved "
        "the gate can still refuse them. Could-not-check, not a pass:\n  "
        + "\n  ".join(samples.unsamplable)
    )
    assert not samples.mismatched, (
        "these recon patterns do not match the value synthesised for them, so "
        "the control would have been vacuous:\n  " + "\n  ".join(samples.mismatched)
    )
    assert len(samples.lines) >= _RECON_FLOOR, (
        f"only {len(samples.lines)} patterns were exercised, below the "
        f"{_RECON_FLOOR} the denylist carried when this was measured."
    )

    planted = tmp_path / "planted"
    planted.mkdir()
    for number, (pat, line) in enumerate(sorted(samples.lines.items())):
        (planted / f"probe_{number:03d}.txt").write_text(line + "\n", encoding="utf-8")

    result = _run_gate(planted)
    assert result.returncode != 0, (
        "the leak gate ACCEPTED a tree holding one matching sample for every "
        "pattern in its own recon denylist. Nothing it lists can refuse "
        f"anything.\n{result.stdout}\n{result.stderr}"
    )
    report = result.stdout + result.stderr
    silent = [pat for pat in samples.lines if pat not in report]
    assert not silent, (
        "the gate refused, but its report never names these patterns, so they "
        "are not what refused and nothing here shows they can:\n  " + "\n  ".join(silent)
    )


@needs_gitleaks
def test_the_real_gate_still_accepts_the_librarys_own_published_sources(tmp_path: Path) -> None:
    """The other half of the control: the gate is not simply refusing everything.

    Without this, a gate wedged into permanent refusal would satisfy the test
    above and block every release while looking rigorous.
    """
    clean = tmp_path / "clean"
    copied = _benign_tree(clean)
    assert copied >= 20, (
        f"only {copied} published sources were copied into the control tree, "
        "so a clean verdict over it would be a verdict about almost nothing."
    )

    result = _run_gate(clean)
    assert result.returncode == 0, (
        "the leak gate REFUSED a tree made only of this library's own published "
        f"sources, so its refusals say nothing.\n{result.stdout}\n{result.stderr}"
    )
    assert "PASSED" in result.stdout, (
        "the gate exited 0 without reporting a pass over the control tree, so "
        f"it is unclear what it did.\n{result.stdout}\n{result.stderr}"
    )
