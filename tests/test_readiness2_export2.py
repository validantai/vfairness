"""Three gaps the export's leak gate and admission manifest do not cover.

The gate refuses every planted secret it is given and the emitted tree matches
its committed manifest exactly, and none of that touched any of these:

1. THE PRIVATE DEVELOPMENT REPO'S OWN NAME reached the public tree. Measured
   2026-09-10 on the tree ``scripts/export-vfairness-to-public.sh --emit-tree``
   emits at HEAD: the ``owner/name`` slug of the repository this library is
   developed in appeared four times in ``tests/test_release_mechanics.py``,
   which named it in order to assert that no release workflow is guarded on it.
   The assertions were right; spelling the slug out was the leak, and the gate
   let it through because the token was not on the denylist. It is now, and
   those four sites assert on the CLASS ("every repository a guard names must be
   the publishing one") instead of on one forbidden instance.

2. AN EXPORTED SCRIPT WAS BROKEN BY THE EXCLUSION LIST.
   the pulse universality sweep script imports
   ``pulse_bias_recall_harness``, which is on ``EXCLUDES`` because it hard-codes
   a path into a private checkout. Run from the emitted tree the sweep died at
   import with a bare ``ModuleNotFoundError`` traceback before argparse ran. It
   now says what is missing, that it is missing on purpose, and that nothing was
   graded, and exits 3.

3. AND THE SAME SHAPE INSIDE IT: a variant whose grading RAISED was recorded as
   ``col_hit=False, verdict_hit=False, link_hit=False`` and folded into every
   denominator, so a crash lowered the recall score exactly like a real miss.
   Measured on a corpus of one unreadable and one readable variant, the old code
   printed ``COL 0/2  VERDICT 0/2  LINK 0/1`` and exited 0. The 1 in ``LINK 0/1``
   was invented outright: that variant carries no proxy mechanism, so its link
   axis is not applicable at all. Ungraded variants are held out of every
   denominator now, reported by name, and turn the run's exit code into 3.

THIS FILE IS SAFE TO PUBLISH, and that is a constraint rather than luck. It
never writes the private slug down: the probe is derived at run time from this
checkout's own git origin, which in the public repo is the public repo. The
gate sections skip there anyway, because the export script exists only in the
private monorepo.

THREE STATES. Where a probe cannot be built (no git, no remote, no denylist to
parse) these tests FAIL saying so. A control that could not be constructed is
could-not-check, and could-not-check is never allowed to read as a pass.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"
SWEEP = LIB_ROOT / "scripts" / "pulse_universality_sweep.py"
HARNESS_MODULE = "pulse_bias_recall_harness"
HARNESS = LIB_ROOT / "scripts" / f"{HARNESS_MODULE}.py"
RECRUITMENT_FIXTURE = Path(__file__).parent / "fixtures" / "recruitment_fairness_dataset.csv"

# Same reasoning as the private export-boundary module and
# tests/test_readiness_export.py: the export script lives only in the private
# monorepo, so in the public repo there is no gate to execute. Skipping is the
# honest answer there; failing would train the next reader to ignore a red
# publish-boundary test.
monorepo_only = pytest.mark.skipif(
    not EXPORT_SCRIPT.is_file(),
    reason=(
        "scripts/export-vfairness-to-public.sh is not present. It exists only in "
        "the private monorepo, which is the only place an export can be run from."
    ),
)

_HAS_GITLEAKS = shutil.which("gitleaks") is not None


# ---------------------------------------------------------------------------
# Probes derived at run time, never written down
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _script_array(name: str) -> tuple[str, ...]:
    """The entries of a one-per-line bash array in the export script.

    Parsed rather than restated, so a list this file checks cannot go on being
    checked after the script stopped using it. Deliberately duplicated instead
    of imported from a sibling test module: this one has to keep working
    whatever happens to the others.
    """
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


def _slug_from_url(url: str) -> str | None:
    """``owner/name`` out of an ssh or https git remote URL."""
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url.strip())
    return match.group(1) if match else None


@functools.lru_cache(maxsize=None)
def _development_repo_slug() -> str:
    """The slug of the repository this checkout came from.

    Read from git, never typed here: this module is exported, and the point of
    the section below is that the development repo's name must not appear in a
    published file, starting with this one.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stdout.strip(), (
        "could not read this checkout's origin remote, so the refusal probe below "
        "cannot be built. That is could-not-check, not a pass:\n"
        f"{proc.stdout}{proc.stderr}"
    )
    slug = _slug_from_url(proc.stdout)
    assert slug, f"could not parse an owner/name slug out of {proc.stdout.strip()!r}"
    return slug


@functools.lru_cache(maxsize=None)
def _publish_target_slug() -> str:
    """The PUBLIC repository the export pushes to, read from the script.

    The over-correction control needs a real, measured neighbour of the token
    under test rather than any old string: this is the slug that legitimately
    appears all over the export tree and must keep passing the gate.
    """
    src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r'^PUBLIC_REPO_URL="\$\{PUBLIC_REPO_URL:-([^}"]+)\}"', src, re.M)
    assert match, f"could not read the default PUBLIC_REPO_URL out of {EXPORT_SCRIPT}"
    slug = _slug_from_url(match.group(1))
    assert slug, f"could not parse an owner/name slug out of {match.group(1)!r}"
    return slug


def _gate(directory: Path) -> subprocess.CompletedProcess:
    """Run the export script's real leak gate over a directory."""
    return subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--gate-only", str(directory)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _denylist_fired(result: subprocess.CompletedProcess) -> bool:
    return "leak gate matched internal/secret tokens" in result.stderr


def _tree_with(tmp_path: Path, label: str, line: str) -> Path:
    directory = tmp_path / label
    directory.mkdir(parents=True)
    (directory / "note.md").write_text(f"# notes\n\n{line}\n", encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# 1. The private development repo's own name
# ---------------------------------------------------------------------------


@monorepo_only
class TestTheDevelopmentRepositoryNameIsRefused:
    """The gate had no pattern for the one string that names the private repo."""

    def test_the_gate_refuses_this_checkouts_own_repository_slug(self, tmp_path):
        # REFUSAL PIN. Before 2026-09-10 this planted line passed the whole gate,
        # denylist and gitleaks alike: recon is not credential-shaped, so no
        # scanner will ever flag it, and it was not on the fixed list.
        slug = _development_repo_slug()
        result = _gate(_tree_with(tmp_path, "private", f"cloned from github.com/{slug}"))

        assert result.returncode != 0, (
            "the leak gate PASSED a tree naming the development repository, which is "
            f"the recon class the denylist exists for.\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert _denylist_fired(result), (
            "the run failed, but not on the denylist, so it proves nothing about the "
            f"pattern under test. stderr:\n{result.stderr}"
        )
        assert "note.md" in result.stderr, (
            "the abort does not name the offending file, so an operator cannot act on it:"
            f"\n{result.stderr}"
        )

    def test_the_gate_does_not_refuse_the_repository_it_publishes_to(self, tmp_path):
        # OVER-CORRECTION CONTROL, on a MEASURED value: the publish target read
        # out of the script itself. It shares an owner with the token above and
        # occurs throughout the real export tree (CITATION.cff, pyproject.toml,
        # both release workflows), so a pattern broad enough to catch it would
        # abort every export on the library's own metadata.
        if not _HAS_GITLEAKS:
            pytest.fail(
                "gitleaks is required to prove the PASS arm: without it the gate aborts "
                "on the missing scanner and a refusal would prove nothing. Install it "
                "(brew install gitleaks). This is could-not-check, not a pass."
            )
        target = _publish_target_slug()
        result = _gate(_tree_with(tmp_path, "public", f"published at github.com/{target}"))

        assert not _denylist_fired(result), (
            f"the denylist refused {target}, which is the repository this library is "
            f"published to. stderr:\n{result.stderr}"
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_two_slugs_are_actually_different(self):
        # Without this the control above could be re-planting the very token the
        # refusal test plants, and both would still "pass".
        assert _development_repo_slug() != _publish_target_slug(), (
            "this checkout's origin IS the publish target, so the refusal test and its "
            "control are planting the same string and neither means anything"
        )

    def test_the_denylist_carries_a_pattern_for_the_development_repository(self):
        # The gate is a fixed list, so pin the entry itself: an accidental
        # deletion would otherwise show up only as the refusal test going green
        # for the wrong reason on some future tree.
        slug = _development_repo_slug()
        matching = [
            pattern
            for pattern in _script_array("LEAK_PATTERNS_RECON")
            if re.search(pattern, slug, re.IGNORECASE)
        ]
        assert matching, (
            f"no entry in LEAK_PATTERNS_RECON matches this checkout's own repository "
            f"slug, so naming the private development repo in a shipped file would "
            f"reach the public tree again. Patterns: {_script_array('LEAK_PATTERNS_RECON')}"
        )


@monorepo_only
class TestNoShippingSourceNamesTheDevelopmentRepository:
    """The gate reads the emitted tree; this reads the sources that build it.

    Same defect, one step earlier, so it is catchable before an export is ever
    attempted. Enumerated from ``git ls-files`` minus the script's own EXCLUDES,
    which is exactly the set ``git archive`` would carry.
    """

    @staticmethod
    def _shipping_files() -> list[Path]:
        listed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--", LIB_ROOT.name],
            capture_output=True,
            text=True,
        )
        assert listed.returncode == 0, (
            f"could not enumerate tracked files, so nothing below was scanned:\n"
            f"{listed.stdout}{listed.stderr}"
        )
        prefix = f"{LIB_ROOT.name}/"
        excludes = _script_array("EXCLUDES")
        globs = _script_array("EXCLUDE_GLOBS")
        files = []
        for line in listed.stdout.splitlines():
            if not line.startswith(prefix):
                continue
            relative = line[len(prefix) :]
            if any(relative == e or relative.startswith(f"{e}/") for e in excludes):
                continue
            name = relative.rsplit("/", 1)[-1]
            if any(name == g or (g.startswith("*") and name.endswith(g[1:])) for g in globs):
                continue
            path = LIB_ROOT / relative
            if path.is_file():
                files.append(path)
        return files

    @staticmethod
    def _occurrences(files: list[Path], token: str) -> list[str]:
        needle = token.lower()
        found = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable; the gate's -I skips these too
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    found.append(f"{path.relative_to(LIB_ROOT)}:{number}")
        return found

    def test_the_file_set_being_scanned_is_not_empty(self):
        # A scan that read nothing reports the same clean result as a clean tree.
        files = self._shipping_files()
        assert len(files) > 400, (
            f"only {len(files)} shipping files were enumerated; the export tree holds "
            "hundreds, so this scan is looking at the wrong thing"
        )

    def test_no_shipping_file_names_the_development_repository(self):
        # REFUSAL PIN, at source level. Four sites in
        # tests/test_release_mechanics.py failed this on 2026-09-10.
        hits = self._occurrences(self._shipping_files(), _development_repo_slug())
        assert hits == [], (
            "these files name the private development repository and are exported to a "
            f"public repo: {hits}. Say what you mean without the slug, the way "
            "NOT_THE_PUBLIC_REPO in tests/test_release_mechanics.py does."
        )

    def test_the_publish_target_is_still_named_where_it_must_be(self):
        # OVER-CORRECTION CONTROL, on MEASURED counts rather than "some hits".
        # 164 occurrences across 30 files, measured 2026-09-10. The floor is well
        # under that so ordinary editing does not trip it, and well over zero so
        # a scanner that stopped reading files cannot pass the test above.
        hits = self._occurrences(self._shipping_files(), _publish_target_slug())
        assert len(hits) >= 100, (
            f"only {len(hits)} occurrences of the publish target were found across the "
            "shipping files, where 164 were measured. Either the library stopped naming "
            "the repository it publishes to, or this scan is no longer reading anything"
        )


# ---------------------------------------------------------------------------
# The first push, which aborts, and did so undocumented
# ---------------------------------------------------------------------------

_SHELL_HELPERS = textwrap.dedent(
    """
    set -euo pipefail
    say() { printf '%s\\n' "$*"; }
    die() { printf 'ABORT: %s\\n' "$*" >&2; exit 1; }
    """
)


def _hermetic_git_env(home: Path, **overrides: str) -> dict[str, str]:
    git = shutil.which("git")
    assert git is not None, "git is required to exercise the export script's guards"
    return {
        "PATH": f"{Path(git).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(home / "no-such-gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        **overrides,
    }


def _git(cwd: Path, *args: str, home: Path) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.email=export2-test@example.invalid",
            "-c",
            "user.name=Export2 Test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_hermetic_git_env(home),
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed:\n{proc.stdout}{proc.stderr}"
    return proc


def _export_block(name: str) -> str:
    """One named block of the export script, extracted verbatim.

    Same markers test_release_mechanics.py uses, and a rename fails loudly here
    rather than silently testing nothing.
    """
    src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^# --- BEGIN {re.escape(name)}\b[^\n]*\n(?P<body>.*?)^# --- END {re.escape(name)} ---",
        src,
        re.S | re.M,
    )
    assert match is not None, (
        f"{EXPORT_SCRIPT} carries no `{name}` block; the guard was removed or its "
        "markers were renamed, and neither is a pass"
    )
    return match.group("body")


def _run_export_block(name: str, home: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", _SHELL_HELPERS + _export_block(name)],
        capture_output=True,
        text=True,
        cwd=str(home),
        env=_hermetic_git_env(home, **env),
    )


@monorepo_only
class TestTheFirstPushIntoTheSeededPublicRepository:
    """The deletion guard's comment assumed the first export finds an EMPTY repo.

    It does not. The publish target was seeded by hand on 2026-08-23 with LICENSE
    and README.md under the subject "Add Apache License 2.0" (read off the live
    repository 2026-09-10), and the tip-subject guard accepts only an empty
    history or a subject beginning "Release v". So the very first --push aborts,
    correctly, on a case the script documented as impossible.

    Nothing here weakens the guard. The fix was to WRITE THE CASE DOWN, and these
    execute the documented procedure end to end so the note cannot drift away
    from what the guard does.
    """

    @staticmethod
    def _seeded_public_repo(tmp_path: Path, subject: str) -> Path:
        public = tmp_path / "public"
        public.mkdir()
        _git(public, "init", home=tmp_path)
        (public / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
        (public / "README.md").write_text("# vfairness\n", encoding="utf-8")
        _git(public, "add", "-A", home=tmp_path)
        _git(public, "commit", "-m", subject, home=tmp_path)
        # Exactly what the script does before the guard runs.
        for entry in public.iterdir():
            if entry.name == ".git":
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        for name, content in {
            "LICENSE": "Apache License 2.0\n",
            "README.md": "# vfairness\n\nthe exported readme\n",
            "pyproject.toml": '[project]\nname = "vfairness"\n',
        }.items():
            (public / name).write_text(content, encoding="utf-8")
        _git(public, "add", "-A", home=tmp_path)
        return public

    def test_the_hand_seeded_tip_is_refused_without_the_flag(self, tmp_path):
        # REFUSAL PIN. This is what the operator meets on the first run, and it
        # deletes nothing, so the deletion list cannot see it: the seeded README
        # stages as a plain M.
        public = self._seeded_public_repo(tmp_path, "Add Apache License 2.0")
        result = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert result.returncode != 0, (
            "the first export overwrote a hand-written public README without asking:\n"
            + result.stdout
            + result.stderr
        )
        assert "Add Apache License 2.0" in result.stderr, result.stderr
        assert "D\t" not in result.stdout, (
            "a deletion was staged after all, so this no longer reproduces the "
            f"first-push case:\n{result.stdout}"
        )

    def test_the_documented_flag_is_what_lets_the_first_push_through(self, tmp_path):
        # OVER-CORRECTION CONTROL, and the documented procedure executed: the
        # header tells the operator to re-run with --accept-deletions once. If
        # that did not work the note would be worse than no note.
        public = self._seeded_public_repo(tmp_path, "Add Apache License 2.0")
        result = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="1"
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "M\tREADME.md" in result.stdout, (
            f"the staged diff the operator is told to read is not printed:\n{result.stdout}"
        )

    def test_a_later_release_still_needs_no_flag(self, tmp_path):
        # CONTROL: the guard must not have become "refuse every export". Once the
        # tip is this script's own commit, --push alone goes through.
        public = self._seeded_public_repo(tmp_path, "Release v0.1.0")
        result = _run_export_block(
            "deletion guard", tmp_path, PUB_DIR=str(public), ACCEPT_DELETIONS="0"
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_script_tells_the_operator_before_they_hit_it(self):
        # DOCUMENTATION PIN, and labelled as one: it proves the note exists, not
        # that the guard works. The three tests above are what prove that.
        src = EXPORT_SCRIPT.read_text(encoding="utf-8")
        # In the HEADER, where usage is read, not merely somewhere in the file:
        # a note buried beside the guard is found only by someone who has already
        # lost the afternoon. Measured against the pre-fix script, all three of
        # these were absent.
        header = src.split("set -euo pipefail", 1)[0]
        assert "FIRST EVER PUSH" in header, (
            "the header no longer warns that the first push aborts, so the next "
            "operator debugs it live"
        )
        assert "--accept-deletions" in header, (
            "the header names the first-push case without naming the flag that resolves it"
        )
        assert "Add Apache License 2.0" in header, (
            "the header does not quote the tip subject the guard actually refuses on, "
            "which is the one string an operator can match against the abort in front of them"
        )


# ---------------------------------------------------------------------------
# 2. The exported sweep, whose only dependency does not ship
# ---------------------------------------------------------------------------


def _run_sweep(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
    }
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )


class TestTheSweepDegradesInsteadOfDying:
    """A shipped script that dies on an ImportError tells a reader nothing."""

    @staticmethod
    def _exported_copy(tmp_path: Path) -> Path:
        """The sweep exactly as the export ships it: alone, no harness beside it."""
        scripts = tmp_path / "scripts"
        scripts.mkdir(parents=True)
        copy = scripts / SWEEP.name
        shutil.copy2(SWEEP, copy)
        assert copy.read_bytes() == SWEEP.read_bytes(), "the copy under test drifted"
        assert not (scripts / HARNESS.name).exists()
        return copy

    def test_the_exported_copy_reports_could_not_check_and_exits_three(self, tmp_path):
        # REFUSAL PIN. Reproduced against the emitted tree before the fix:
        # ModuleNotFoundError, exit 1, nothing printed but a traceback.
        copy = self._exported_copy(tmp_path)
        result = _run_sweep(copy, "--corpus", str(tmp_path / "no-corpus"), cwd=tmp_path)

        assert result.returncode == 3, (
            "the exported sweep did not report could-not-check (exit 3):\n"
            f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Traceback" not in result.stderr, f"it still dies with a traceback:\n{result.stderr}"
        assert HARNESS_MODULE in result.stderr, (
            f"the message does not name what is missing:\n{result.stderr}"
        )
        assert "COULD NOT CHECK" in result.stderr, result.stderr
        # And it must not read as a measurement of zero, which is the whole point.
        assert "not a score of zero" in result.stderr, result.stderr

    def test_it_does_not_print_a_score_it_did_not_compute(self, tmp_path):
        copy = self._exported_copy(tmp_path)
        result = _run_sweep(copy, "--corpus", str(tmp_path / "no-corpus"), cwd=tmp_path)
        combined = result.stdout + result.stderr
        assert "UNIVERSALITY SCORE" not in combined, (
            f"a sweep that never ran printed a score line:\n{combined}"
        )

    @pytest.mark.skipif(
        not HARNESS.is_file(),
        reason="the grading harness ships only in the development repo",
    )
    def test_the_harness_being_present_does_not_take_the_degraded_path(self, tmp_path):
        # OVER-CORRECTION CONTROL, on a MEASURED exit code. A degradation that
        # fired whenever anything was wrong would pass the pin above and break
        # the script for the one repository it works in. Exit 2 is the distinct
        # "no corpus" verdict, and it must still be reachable.
        corpus = tmp_path / "empty-corpus"
        corpus.mkdir()
        result = _run_sweep(SWEEP, "--corpus", str(corpus), cwd=LIB_ROOT)

        assert result.returncode == 2, (
            "with the harness importable the sweep no longer reaches its own "
            f"no-corpus verdict:\nrc={result.returncode}\n{result.stdout}{result.stderr}"
        )
        assert "no L*.csv files found" in result.stderr, result.stderr
        assert "COULD NOT CHECK" not in result.stderr + result.stdout, (
            "the harness is importable, so nothing here is could-not-check:\n"
            f"{result.stdout}{result.stderr}"
        )

    @monorepo_only
    def test_the_exclusion_that_makes_this_necessary_is_still_in_place(self):
        # If the harness is ever admitted to the export, the degradation becomes
        # dead code and this whole section should be re-decided rather than left
        # standing on a premise that stopped being true.
        excludes = _script_array("EXCLUDES")
        assert f"scripts/{HARNESS.name}" in excludes, (
            f"scripts/{HARNESS.name} is no longer on EXCLUDES: the sweep's degraded "
            f"path is now unreachable. Patterns: {excludes}"
        )
        assert f"scripts/{SWEEP.name}" not in excludes, (
            "the sweep itself is excluded now, so it no longer ships and the "
            "degradation it grew is pointless"
        )


# ---------------------------------------------------------------------------
# 3. A variant nobody could grade is not a variant that failed
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _sweep_module():
    """The sweep, imported for its pure summary function.

    Importing the real file rather than a copy: pyproject pins pytest's
    pythonpath, and a copy on a hand-built path is not what anything runs.
    """
    spec = importlib.util.spec_from_file_location("_export2_sweep_under_test", SWEEP)
    assert spec is not None and spec.loader is not None, f"cannot load {SWEEP}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAnUngradedVariantIsHeldOutOfEveryScore:
    """The defect class, inside the script: a crash counted as a detection miss."""

    def test_a_variant_that_raised_lands_in_no_denominator(self):
        # REFUSAL PIN. The old record was col_hit/verdict_hit/link_hit all False,
        # which put the variant in the COL denominator AND in the verdict and
        # link denominators, none of which anyone measured.
        summary = _sweep_module()._summarise(
            [
                {
                    "variant": "L1_gender_in_tech",
                    "error": "No columns to parse from file",
                    "col_hit": None,
                    "verdict_hit": None,
                    "link_hit": None,
                },
                {"variant": "L2_race", "col_hit": True, "verdict_hit": True, "link_hit": None},
            ]
        )
        assert summary["ungraded"] == 1, summary
        assert summary["graded"] == 1, summary
        assert summary["col_pass"] == 1, summary
        assert summary["v_relevant"] == 1, summary
        assert summary["v_pass"] == 1, summary
        # The invented one. L1 carries a direct mechanism, so its link axis is
        # not applicable at all; the old code still counted a link denominator
        # of 1 for it and reported LINK 0/1.
        assert summary["l_relevant"] == 0, summary
        assert summary["l_pass"] == 0, summary

    def test_measured_hits_and_misses_are_still_counted_exactly(self):
        # OVER-CORRECTION CONTROL, on MEASURED values rather than "some number".
        # A summariser that dropped every variant would pass the pin above and
        # report a perfect, empty sweep.
        summary = _sweep_module()._summarise(
            [
                {"variant": "a", "col_hit": True, "verdict_hit": True, "link_hit": None},
                {"variant": "b", "col_hit": True, "verdict_hit": False, "link_hit": None},
                {"variant": "c", "col_hit": False, "verdict_hit": None, "link_hit": True},
                {"variant": "d", "col_hit": False, "verdict_hit": None, "link_hit": False},
            ]
        )
        assert summary == {
            "graded": 4,
            "ungraded": 0,
            "col_pass": 2,
            "v_relevant": 2,
            "v_pass": 1,
            "l_relevant": 2,
            "l_pass": 1,
        }, summary

    @pytest.mark.skipif(
        not HARNESS.is_file(),
        reason="the grading harness ships only in the development repo",
    )
    def test_end_to_end_one_unreadable_variant_does_not_lower_the_score(self, tmp_path):
        # The same thing through the real script, because the summary function
        # being right proves nothing about what main() prints and returns.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "L1_gender_in_tech.csv").write_text("", encoding="utf-8")
        (corpus / "L2_race.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        result = _run_sweep(SWEEP, "--corpus", str(corpus), cwd=LIB_ROOT)

        assert result.returncode == 3, (
            "a sweep with an ungraded variant reported a clean exit:\n"
            f"rc={result.returncode}\n{result.stdout}{result.stderr}"
        )
        assert "COL 0/1" in result.stdout, (
            "the unreadable variant is still in the COL denominator (it read COL 0/2 "
            f"before the fix):\n{result.stdout}"
        )
        assert "LINK 0/0" in result.stdout, (
            f"a link denominator was invented for a variant nobody graded:\n{result.stdout}"
        )
        assert "COULD NOT CHECK      1/2" in result.stdout, result.stdout
        assert "L1_gender_in_tech" in result.stdout, (
            f"the ungraded variant is not named, so nobody can go and look:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# The shipped recruitment fixture: synthetic, established by measurement
# ---------------------------------------------------------------------------


class TestTheRecruitmentFixtureIsProvablySynthetic:
    """2700 rows of names, emails, phones, dates of birth and street addresses.

    Nothing in or beside the file says it is generated, so a reader has to take
    that on trust. This does not add the note that is still missing; it makes
    the claim CHECKABLE, and it goes red the day someone swaps real data in.

    The markers are not stylistic. A per-row planted-bias answer key and a
    ground-truth qualification score are things no real recruitment dataset has,
    and cdn.example.com is the domain RFC 2606 reserves for exactly this.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _rows() -> tuple[dict[str, str], ...]:
        with RECRUITMENT_FIXTURE.open(newline="", encoding="utf-8") as handle:
            return tuple(csv.DictReader(handle))

    def test_every_row_carries_a_planted_bias_answer_key(self):
        rows = self._rows()
        assert len(rows) == 2700, len(rows)
        assert all("_bias_flags" in row for row in rows), (
            "the answer-key column is gone, which is the strongest single piece of "
            "evidence that this file is generated rather than collected"
        )
        assert all(row["true_qualification_score"] for row in rows), (
            "a ground-truth qualification score is missing on some rows; nobody has "
            "that for real applicants, and it is why this fixture can be graded"
        )
        planted = {flag for row in rows for flag in (row["_bias_flags"] or "").split(",") if flag}
        # The alphabet as MEASURED on 2026-09-10, not as the harness docstring
        # describes it. B5 is signed (B5+/B5-, a university-tier advantage and
        # its mirror) and B9, the photo channel, carries no per-row tag at all:
        # it is planted through photo_attractiveness_score. Pinned to what is
        # actually in the file so a regenerated fixture that quietly dropped a
        # mechanism shows up here.
        assert planted == {"B1", "B2", "B3", "B4", "B5+", "B5-", "B6", "B7", "B8"}, sorted(planted)
        # And the column is an answer key rather than a constant: 195 of the
        # 2700 rows carry no planted mechanism, which is what makes the graded
        # variants mean anything.
        unflagged = sum(1 for row in rows if not (row["_bias_flags"] or "").strip())
        assert 50 < unflagged < 600, unflagged

    def test_every_contactable_field_points_at_a_reserved_example_host(self):
        rows = self._rows()
        assert all(row["photo_url"].startswith("https://cdn.example.com/") for row in rows), (
            "a photo URL points somewhere real"
        )

    def test_the_identity_fields_are_drawn_from_a_small_generator_pool(self):
        # MEASURED, not "looks synthetic": 2700 rows built out of 106 first
        # names, 57 surnames, 14 cities and 24 street names is a generator, not
        # a cohort. Bounds are loose enough to survive a regenerated fixture and
        # tight enough that a real extract could not pass.
        rows = self._rows()
        assert len({row["first_name"] for row in rows}) < 400, "first-name pool"
        assert len({row["last_name"] for row in rows}) < 300, "surname pool"
        assert len({row["city"] for row in rows}) < 60, "city pool"
        streets = {re.sub(r"^\d+\s+", "", row["street_address"]) for row in rows}
        assert len(streets) < 120, f"{len(streets)} distinct street names"
