"""The export must not ship a file that REBUILDS an internal token at run time.

``scripts/export-vfairness-to-public.sh`` vets the export tree with ``grep``, so
it can only ever see tokens that are present as LITERALS. A file that assembles
its strings at run time is invisible to it by construction, and the export tree
carried exactly such a file: ``tests/test_export_boundary.py`` builds every one
of its probes from fragments, deliberately, so that the gate does not abort on
its own test suite.

Measured 2026-09-09 against the tree the script emits at HEAD: importing the
EXPORTED copy of that module and printing its ``_MUST_REFUSE`` list reproduced
the developer home directory and bare username, the workstation name, the
internal LAN engine host together with its ssh user, the private platform
checkout and its environment variable, that repo's internal directory layout,
the private consumer deployment path, a platform source filename, and both
secret stores in use. Every one of those is an entry in the gate's own
``LEAK_PATTERNS_RECON``, and that file was the only one in the 778-file tree
matching the LAN-subnet pattern. It also skipped itself entirely over there (37
tests, 37 skips, because its ``pytestmark`` needs the export script, which
exists only in the private monorepo), so publishing it bought nothing at all.

The fix is one line in ``EXCLUDES``. This module pins BOTH halves:

* the instance: that path must not reach the export tree, and the reason it does
  not must be the exclude entry rather than the file having quietly vanished;
* the class: NO exported file may reassemble a denylisted token, whether by
  concatenation, ``str.join``, ``chr()`` arithmetic, ``.upper()``/``.capitalize()``
  case surgery, an f-string, or a base64 blob. Fixing one file is not fixing the
  class, and the gate cannot grow a pattern for this: there is nothing to grep.

THIS FILE IS SAFE TO PUBLISH, and that is a design constraint rather than luck.
It hard-codes no internal token and no probe. Every value it tests with is
derived at run time from the live denylist in the export script, which does not
exist in the public repo, so there is nothing here to reassemble. Verified by
running this module's own detector over this module.

THREE STATES. A file that cannot be decoded, parsed or read is could-not-check,
never a pass: those files are collected separately and asserted empty, and the
scan's coverage counters are asserted against measured floors so that a scan
which silently read nothing cannot report a clean tree.
"""

from __future__ import annotations

import ast
import base64
import functools
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"

#: The instance that prompted this module. A plain path, not a token.
BOUNDARY_TEST = "tests/test_export_boundary.py"

#: Suffixes the scan is allowed to leave unread. Anything else that reads as
#: binary is could-not-check and must be reported, not skipped quietly.
_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".ttf"}

# Same reasoning as tests/test_export_boundary.py: the export script lives only
# in the private monorepo, so in the public repo there is no export to inspect
# and no denylist to derive probes from. Skipping is the honest answer; failing
# would train the next reader to ignore a red publish-boundary test.
pytestmark = pytest.mark.skipif(
    not EXPORT_SCRIPT.is_file(),
    reason=(
        "scripts/export-vfairness-to-public.sh is not present. It exists only in "
        "the private monorepo, which is the only place an export can be run from."
    ),
)


# ---------------------------------------------------------------------------
# Reading the live denylist and exclude list out of the shipped script
# ---------------------------------------------------------------------------
# Parsed from the script rather than restated here, for the same reason
# test_export_boundary.py parses it: a copy of the list would go on being
# checked long after the script stopped using it. The parser is deliberately
# duplicated instead of imported, so this module keeps working whatever happens
# to the module it exists to keep out of the export.


@functools.lru_cache(maxsize=None)
def _script_array(name: str) -> tuple[str, ...]:
    # Cached because the scan asks for the denylist once per file per half, and
    # re-reading and re-compiling a 33KB script 1500 times turned a two-second
    # scan into thirty. Nothing edits the script mid-run.
    src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    opened = re.search(rf"^{name}=\(", src, re.M)
    assert opened, f"could not find the {name} array in {EXPORT_SCRIPT}"
    rest = src[opened.end() :]

    # Both array styles are in this script: one entry per line closed by a bare
    # ")", and the whole array on one line.
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
def _denylist() -> tuple[tuple[str, str, re.Pattern[str]], ...]:
    """(half, pattern source, compiled) for every denylist entry, with the
    script's own casing rules: recon case-insensitive, secrets case-sensitive."""
    out: list[tuple[str, str, re.Pattern[str]]] = []
    for pat in _script_array("LEAK_PATTERNS_RECON"):
        out.append(("recon", pat, re.compile(pat, re.IGNORECASE)))
    for pat in _script_array("LEAK_PATTERNS_SECRET"):
        out.append(("secret", pat, re.compile(pat)))
    return tuple(out)


def _denylist_hits(text: str) -> list[tuple[str, str]]:
    """(pattern source, matched text) for every denylist entry this text trips."""
    hits = []
    for _half, pat, rx in _denylist():
        found = rx.search(text)
        if found:
            hits.append((pat, found.group(0)))
    return hits


# ---------------------------------------------------------------------------
# The detector: what does this file's code EVALUATE to?
# ---------------------------------------------------------------------------

_STR_METHODS = {
    "upper",
    "lower",
    "capitalize",
    "title",
    "casefold",
    "swapcase",
    "strip",
    "lstrip",
    "rstrip",
    "replace",
    "format",
}
_B64_DECODERS = ("b64decode", "b32decode", "b16decode", "b85decode", "a85decode")


def _fold_strings(source: str) -> list[str]:
    """Every string this module's code can be constant-folded down to.

    Deliberately NOT a list of the file's string literals: the gate already sees
    those. This resolves module-level names, ``+`` concatenation, f-strings,
    ``str.join``, the case-changing methods, ``chr()``, ``repr()`` and base64
    decodes, which is the set of moves that hides a token from a grep.

    Raises ``SyntaxError`` so an unparseable file becomes could-not-check rather
    than an empty (clean-looking) result.
    """
    tree = ast.parse(source)
    env: dict[str, str] = {}

    def ev(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, bytes):
                try:
                    return node.value.decode("utf-8")
                except UnicodeDecodeError:
                    return None
            return None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = ev(node.left), ev(node.right)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            base = ev(node.left)
            count = node.right
            if base is not None and isinstance(count, ast.Constant):
                if isinstance(count.value, int) and 0 < count.value <= 64:
                    return base * count.value
            return None
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for piece in node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(piece.value)
                elif isinstance(piece, ast.FormattedValue):
                    inner = ev(piece.value)
                    if inner is None:
                        return None
                    parts.append(inner)
                else:
                    return None
            return "".join(parts)
        if isinstance(node, ast.Call):
            return ev_call(node)
        return None

    def ev_call(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == "chr" and node.args:
                point = node.args[0]
                if isinstance(point, ast.Constant) and isinstance(point.value, int):
                    if 0 <= point.value <= 0x10FFFF:
                        return chr(point.value)
                return None
            if func.id in {"repr", "str"} and node.args:
                inner = ev(node.args[0])
                if inner is None:
                    return None
                return repr(inner) if func.id == "repr" else inner
            return None
        if isinstance(func, ast.Attribute):
            if func.attr in _B64_DECODERS and node.args:
                blob = ev(node.args[0])
                if blob is None:
                    return None
                for decoder in (base64.b64decode, base64.b32decode, base64.b16decode):
                    try:
                        return decoder(blob).decode("utf-8")
                    except Exception:  # not that encoding; try the next one
                        continue
                return None
            base = ev(func.value)
            if base is None:
                return None
            if func.attr == "join" and node.args:
                items = node.args[0]
                if isinstance(items, (ast.List, ast.Tuple)):
                    values = [ev(element) for element in items.elts]
                    if all(value is not None for value in values):
                        return base.join([value for value in values if value is not None])
                return None
            if func.attr in _STR_METHODS:
                args = [ev(argument) for argument in node.args]
                if any(argument is None for argument in args):
                    return None
                try:
                    return str(getattr(base, func.attr)(*args))
                except (TypeError, ValueError, IndexError, KeyError):
                    return None
            return None
        return None

    # Two passes over the module-level assignments, so a helper defined after
    # its first use still resolves.
    for _pass in range(2):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value is None:
                    continue
                value = ev(node.value)
                if value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value

    folded = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call, ast.Name)):
            value = ev(node)
            if value:
                folded.append(value)
    folded.extend(env.values())
    return folded


_FRAGMENT = re.compile(r"""(?<!\\)"([^"\n]*)"|(?<!\\)'([^'\n]*)'""")


def _glued_fragments(text: str) -> str:
    """Every quoted fragment in the file, in source order, with the glue removed.

    The crude half of the detector, and the reason it exists: the AST half only
    understands Python. A YAML step, a shell heredoc or a Markdown snippet can
    paste an internal token together just as effectively, and this catches those
    without needing a parser for each language.
    """
    return "".join(double or single for double, single in _FRAGMENT.findall(text))


@dataclass
class _Scan:
    findings: list[str] = field(default_factory=list)
    text_files: int = 0
    python_files: int = 0
    folded_strings: int = 0
    unparsed: list[str] = field(default_factory=list)
    undecodable: list[str] = field(default_factory=list)
    unscanned: list[str] = field(default_factory=list)


def _scan_tree(root: Path) -> _Scan:
    scan = _Scan()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        raw = path.read_bytes()
        if b"\x00" in raw[:8000]:
            scan.unscanned.append(rel)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            scan.undecodable.append(rel)
            continue
        scan.text_files += 1

        if path.suffix == ".py":
            scan.python_files += 1
            try:
                folded = _fold_strings(text)
            except SyntaxError as exc:
                scan.unparsed.append(f"{rel}: {exc}")
                folded = []
            scan.folded_strings += len(folded)
            for value in folded:
                for pattern, matched in _denylist_hits(value):
                    scan.findings.append(f"{rel}: rebuilt {matched!r} (denylist {pattern})")

        for pattern, matched in _denylist_hits(_glued_fragments(text)):
            scan.findings.append(f"{rel}: fragments glue to {matched!r} (denylist {pattern})")
    return scan


# ---------------------------------------------------------------------------
# A control value for every denylist pattern, derived from the pattern itself
# ---------------------------------------------------------------------------
# The positive control cannot be a hard-coded token: writing one here would put
# the leak in the file that exists to stop it, and this module ships. So a
# matching sample is GENERATED from each pattern and checked against the real
# pattern with re.search before it is used. A pattern that cannot be sampled is
# reported as could-not-check and fails the control, never skipped.


def _class_first_char(body: str) -> str | None:
    if body.startswith("^"):
        return None
    if body.startswith("\\") and len(body) > 1:
        return body[1]
    return body[0] if body else None


def _split_alternatives(pattern: str) -> list[str]:
    parts, depth, current = [], 0, []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            current.append(pattern[index : index + 2])
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _sample_from_pattern(pattern: str) -> str | None:
    """A concrete string that the given denylist regex matches, or None."""
    out: list[str] = []
    index, length = 0, len(pattern)
    while index < length:
        char = pattern[index]
        if char == "\\":
            following = pattern[index + 1] if index + 1 < length else ""
            index += 2
            if following in {"b", "B", "A", "Z"}:
                unit = ""  # zero-width assertion
            elif following == "d":
                unit = "7"
            elif following == "w":
                unit = "a"
            elif following == "s":
                unit = " "
            else:
                unit = following
        elif char == "[":
            close = pattern.find("]", index + 1)
            if close == -1:
                return None
            first = _class_first_char(pattern[index + 1 : close])
            if first is None:
                return None
            unit = first
            index = close + 1
        elif char == "(":
            depth, cursor = 1, index + 1
            while cursor < length and depth:
                if pattern[cursor] == "\\":
                    cursor += 2
                    continue
                if pattern[cursor] == "(":
                    depth += 1
                elif pattern[cursor] == ")":
                    depth -= 1
                cursor += 1
            inner = pattern[index + 1 : cursor - 1]
            index = cursor
            if inner.startswith("?:"):
                inner = inner[2:]
            sub = _sample_from_pattern(_split_alternatives(inner)[0])
            if sub is None:
                return None
            unit = sub
        elif char == ".":
            unit, index = "x", index + 1
        elif char in "^$":
            unit, index = "", index + 1
        elif char == "|":
            break  # first top-level alternative is enough
        else:
            unit, index = char, index + 1

        if index < length and pattern[index] in "*+?{":
            quantifier = pattern[index]
            if quantifier == "{":
                close = pattern.find("}", index)
                if close == -1:
                    return None
                low = pattern[index + 1 : close].split(",")[0]
                index = close + 1
                unit = unit * (int(low) if low.isdigit() else 1)
            else:
                index += 1
        out.append(unit)
    return "".join(out)


def _split_invisibly(sample: str) -> tuple[str, str] | None:
    """Cut the sample so that NEITHER half trips the denylist on its own.

    That is what makes a planted control a real control: if a half still matched,
    the detector could be passing on a plain literal that the gate would have
    caught anyway.
    """
    middle = len(sample) // 2
    order = sorted(range(1, len(sample)), key=lambda cut: abs(cut - middle))
    for cut in order:
        head, tail = sample[:cut], sample[cut:]
        if not _denylist_hits(head) and not _denylist_hits(tail):
            return head, tail
    return None


# ---------------------------------------------------------------------------
# The export tree, built by the real script
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def export_tree(tmp_path_factory) -> Path:
    """The tree the export would publish, via the script's read-only mode.

    ``--emit-tree`` writes the tree BEFORE running the leak gate, so this fixture
    works on a machine without gitleaks and on a day the gate legitimately
    refuses HEAD. It never pushes, clones, commits or tags.
    """
    dest = tmp_path_factory.mktemp("readiness-export") / "tree"
    result = subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--emit-tree", str(dest)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert dest.is_dir(), (
        "the export script emitted no tree, so nothing here could be checked.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return dest


@pytest.fixture(scope="module")
def tree_scan(export_tree: Path) -> _Scan:
    return _scan_tree(export_tree)


# ---------------------------------------------------------------------------
# 1.  The instance
# ---------------------------------------------------------------------------


def test_the_export_does_not_publish_the_boundary_test(export_tree: Path) -> None:
    """REFUSAL PIN for the finding, plus the control that makes it mean something.

    Absence alone would also be produced by the file having been renamed or
    deleted, which would leave the class wide open while this test stayed green.
    So the file must still be here, the exclude entry must still name it, and
    only then is its absence from the tree evidence.
    """
    assert (LIB_ROOT / BOUNDARY_TEST).is_file(), (
        f"{BOUNDARY_TEST} is gone from this repo, so its absence from the export "
        "tree proves nothing about EXCLUDES."
    )
    assert BOUNDARY_TEST in _script_array("EXCLUDES"), (
        f"{BOUNDARY_TEST} is not in EXCLUDES. If it is out of the tree today it is "
        "by accident, and the next tracked copy of it ships."
    )
    assert not (export_tree / BOUNDARY_TEST).exists(), (
        f"the export would publish {BOUNDARY_TEST}. Its probes are assembled at "
        "run time, so the leak gate cannot see them, and importing the published "
        "copy reproduces every recon token the gate exists to block."
    )


def test_the_rest_of_the_test_suite_is_still_published(export_tree: Path) -> None:
    """OVER-CORRECTION CONTROL on the exclude entry, with a measured count.

    An exclude entry is a path literal, and a fat-fingered one ("tests/") would
    delete the entire suite from the public repo while every assertion above
    stayed green. The public package must still ship its tests.
    """
    published = sorted(p.name for p in (export_tree / "tests").glob("test_*.py"))

    assert len(published) >= 100, (
        f"the export publishes only {len(published)} test modules; the exclude "
        "entry has taken more than the one file with it."
    )
    assert "test_export_boundary.py" not in published


# ---------------------------------------------------------------------------
# 2.  The class
# ---------------------------------------------------------------------------


def test_no_exported_file_rebuilds_a_denylisted_token(tree_scan: _Scan) -> None:
    """REFUSAL PIN for the class the instance belongs to.

    The gate greps for literals, so this is the half of the boundary it cannot
    cover: anything the export tree assembles at run time. One file did exactly
    that; this asserts that none does.
    """
    assert not tree_scan.findings, (
        "these exported files rebuild a denylisted token at run time, where the "
        "grep-based leak gate cannot see it:\n  " + "\n  ".join(tree_scan.findings)
    )


def test_the_scan_actually_read_the_export_tree(tree_scan: _Scan) -> None:
    """OVER-CORRECTION CONTROL asserting MEASURED coverage, not membership.

    A scan that read nothing reports the same empty finding list as a clean tree.
    Measured on the tree at 2026-09-09: 763 text files, 477 Python modules, 4406
    folded strings, zero unparsed and zero undecodable. The floors sit well below
    those and would still catch a scan that had quietly stopped working.
    """
    assert tree_scan.text_files >= 600, f"only {tree_scan.text_files} text files were read"
    assert tree_scan.python_files >= 400, f"only {tree_scan.python_files} modules were parsed"
    assert tree_scan.folded_strings >= 2000, (
        f"only {tree_scan.folded_strings} strings were folded; the AST half of the "
        "detector is not doing any work"
    )
    # THREE STATES: unreadable is not clean.
    assert not tree_scan.unparsed, f"could not parse, so could not check: {tree_scan.unparsed}"
    assert not tree_scan.undecodable, f"could not decode: {tree_scan.undecodable}"
    stray = [p for p in tree_scan.unscanned if Path(p).suffix.lower() not in _MEDIA_SUFFIXES]
    assert not stray, f"unread non-media files; these were NOT checked at all: {stray}"


def test_the_detector_rebuilds_a_split_token_for_every_denylist_pattern(tmp_path) -> None:
    """POSITIVE CONTROL, generated from the live denylist.

    Without it, a detector that had stopped folding anything would report the
    same clean tree as the fix does. Every pattern in the script gets a sample
    generated from itself, verified against the real regex, then cut so that
    neither half trips the denylist alone, then planted as a concatenation. The
    detector has to put it back together.
    """
    patterns = [pattern for _half, pattern, _rx in _denylist()]
    assert len(patterns) >= 15, f"only {len(patterns)} denylist patterns were parsed"

    missed, unsampled = [], []
    for index, pattern in enumerate(patterns):
        sample = _sample_from_pattern(pattern)
        if sample is None or not re.search(pattern, sample, re.IGNORECASE):
            unsampled.append(pattern)
            continue
        halves = _split_invisibly(sample)
        if halves is None:
            unsampled.append(f"{pattern} (no invisible split)")
            continue
        head, tail = halves
        planted = tmp_path / f"planted_{index}.py"
        planted.write_text(f"VALUE = {head!r} + {tail!r}\n", encoding="utf-8")

        raw = planted.read_text(encoding="utf-8")
        assert not _denylist_hits(raw), (
            f"the planted file for {pattern} trips the denylist as written, so it "
            "proves nothing about reassembly"
        )
        # BOTH halves of the detector are asked separately. The whole-tree scan
        # alone would stay green with the AST half dead, because the fragment
        # glue also catches a plain `+`; a control that cannot tell them apart
        # would report a working detector while half of it did nothing.
        scan = _scan_tree(tmp_path)
        by_scan = any(planted.name in finding for finding in scan.findings)
        by_ast = any(_denylist_hits(value) for value in _fold_strings(raw))
        by_glue = bool(_denylist_hits(_glued_fragments(raw)))
        if not (by_scan and by_ast and by_glue):
            missed.append(f"{pattern} (scan={by_scan}, ast={by_ast}, glue={by_glue})")
        planted.unlink()

    assert not unsampled, (
        f"could not build a control for these patterns, so they are UNCHECKED "
        f"rather than clean: {unsampled}"
    )
    assert not missed, f"the detector did not rebuild a split token for: {missed}"


def test_the_detector_rebuilds_the_other_assembly_styles(tmp_path) -> None:
    """POSITIVE CONTROL for the moves that are not plain ``+``.

    ``str.join`` on a split hostname, ``chr()`` arithmetic, case surgery on a
    fragment, an f-string and a base64 blob all hide a token from a grep just as
    well. The sample is still generated from the live denylist, so nothing here
    is a hard-coded internal string.
    """
    pattern = next(pat for _half, pat, _rx in _denylist() if _sample_from_pattern(pat))
    sample = _sample_from_pattern(pattern)
    assert sample and re.search(pattern, sample, re.IGNORECASE)
    halves = _split_invisibly(sample)
    assert halves is not None, f"no invisible split for {pattern}"
    head, tail = halves
    blob = base64.b64encode(sample.encode("utf-8")).decode("ascii")

    styles = {
        "join": f"VALUE = ''.join([{head!r}, {tail!r}])\n",
        "chr": f"VALUE = chr({ord(sample[0])}) + {sample[1:]!r}\n",
        "case_surgery": f"VALUE = {head.lower()!r}.upper() + {tail!r}\n",
        "fstring": f"HEAD = {head!r}\nVALUE = f'{{HEAD}}{tail}'\n",
        "base64": f"import base64\nVALUE = base64.b64decode({blob!r}).decode()\n",
    }
    blind = []
    for style, source in styles.items():
        planted = tmp_path / f"style_{style}.py"
        planted.write_text(source, encoding="utf-8")
        folded = _fold_strings(source)
        hits = [value for value in folded if _denylist_hits(value)]
        # case surgery only reproduces the token when the pattern is matched
        # case-insensitively, which is the recon half; do not claim otherwise.
        if not hits and not (style == "case_surgery" and head.lower().upper() != head):
            blind.append(style)
        planted.unlink()

    assert not blind, f"the detector is blind to these assembly styles: {blind}"


def test_the_detector_leaves_the_librarys_own_text_alone(tmp_path) -> None:
    """OVER-CORRECTION CONTROL: broader must not mean noisier.

    These four lines are what the export script's own comments promise are
    allowed. A detector that flagged them would push the next operator towards
    switching it off, and this boundary deliberately has no override.
    """
    allowed = "\n".join(
        [
            "blocked = ['192.168.1.10', '10.0.0.5', '169.254.169.254']",
            "Set SUPABASE_SERVICE_ROLE_KEY in the environment to enable it.",
            "base_url = 'http://localhost:11434'",
            "proxy = 'legitimacy is scored per feature'",
        ]
    )
    (tmp_path / "legit.py").write_text(allowed + "\n", encoding="utf-8")

    scan = _scan_tree(tmp_path)

    assert not scan.findings, f"the detector flagged documented-allowed text: {scan.findings}"
    assert scan.folded_strings >= 0 and scan.python_files == 1, (
        f"the control file was not actually scanned: {scan}"
    )


def test_this_module_is_itself_safe_to_publish() -> None:
    """This file ships. It must not be the next instance of the finding.

    Every probe here is derived from the export script at run time, so there
    should be nothing in this source to reassemble and nothing to grep.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    folded = _fold_strings(source)

    rebuilt = [(value, _denylist_hits(value)) for value in folded if _denylist_hits(value)]
    assert not rebuilt, f"this module rebuilds a denylisted token: {rebuilt}"
    assert not _denylist_hits(_glued_fragments(source)), (
        "this module's quoted fragments glue into a denylisted token"
    )
    assert not _denylist_hits(source), "this module carries a denylisted token as a literal"
    # MEASURED, not membership: the folding really ran over this file.
    assert len(folded) >= 10, f"only {len(folded)} strings folded out of this module"
