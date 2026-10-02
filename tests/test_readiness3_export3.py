"""Three things the export tree still disclosed after the repository slug was closed.

The leak gate refused the private repository's ``owner/name`` slug from the day it
was added, and none of the following was touched by that:

1. THE CHECKOUT DIRECTORY IS A SECOND FORM OF THE SAME NAME. The slug entry
   deliberately does not cover the bare token, because a public marketing asset
   carries it, so the private checkout reached the public tree in its PATH form
   instead: ``scripts/regenerate_gallery_examples.py`` told the reader to run it
   with the virtualenv of the private monorepo checkout. Measured 2026-09-10 on
   the tree ``--emit-tree`` writes at HEAD: one match in 785 files. The gate now
   carries the path form, and that docstring names a plain ``.venv`` at the
   repository root, which is a path a reader of the PUBLIC repo can use. The two
   halves are only correct together: adding the pattern alone lands the gate red
   on a file nobody touched, and fixing the docstring alone leaves the next
   occurrence unguarded.

2. SHIPPED FILES POINT AT PATHS THE EXPORT DROPS. Two documentation lists linked
   a slide deck that ``EXCLUDES`` removes, so the link was a guaranteed 404 in
   the public repository. Those are fixed. Others remain and are RECORDED here
   rather than hidden: a shipped script whose closing line tells the reader to
   run an internal harness that is not published, and a shipped document whose
   release step names a script that is not published either. Both are outside
   this change's ownership; the ratchet below stops the list from growing.

3. THE INTERNAL AUDIT INVENTORY WAS DISCLOSED BY NAME. Files under
   ``docs/audits`` are on ``EXCLUDES``, and the exported tree named six of those
   documents anyway. Naming an internal record a reader cannot open tells them an
   assessment exists and denies them its content. Measured 2026-09-10 on the same
   emitted tree: 27 sites in 21 files. Every site in a file this change owns now
   carries the INFORMATION (the date, the finding count, the decision) with the
   unopenable filename dropped; the remainder is recorded below.

NOTHING FORBIDDEN IS WRITTEN DOWN HERE, and that is a constraint rather than
luck. This module ships. The checkout-directory probe is derived at run time from
this checkout's own layout, and the audit document names are read out of the
directory itself, so the file cannot be the thing that leaks them. It is also why
the probes below are built rather than pasted.

THREE STATES. Where a probe cannot be constructed (no export script, no audit
directory, no secret scanner) these tests SKIP or FAIL saying so. A control that
could not be built is could-not-check, and could-not-check never reads as a pass.
"""

from __future__ import annotations

import fnmatch
import functools
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export-vfairness-to-public.sh"
GALLERY_SCRIPT = LIB_ROOT / "scripts" / "regenerate_gallery_examples.py"
AUDIT_DIR = LIB_ROOT / "docs" / "audits"

# Same reasoning as tests/test_readiness_export.py: the export script exists only
# in the private monorepo, so in the public repository there is no gate to run.
# Skipping is the honest answer there; failing would train the next reader to
# ignore a red publish-boundary test.
monorepo_only = pytest.mark.skipif(
    not EXPORT_SCRIPT.is_file(),
    reason=(
        "scripts/export-vfairness-to-public.sh is not present. It exists only in "
        "the private monorepo, which is the only place an export can run from."
    ),
)
audits_present = pytest.mark.skipif(
    not AUDIT_DIR.is_dir(),
    reason=(
        "docs/audits is not part of this checkout. It is on the export script's "
        "EXCLUDES list, so the public repository never receives it."
    ),
)

_HAS_GITLEAKS = shutil.which("gitleaks") is not None
_NEEDS_GITLEAKS = pytest.mark.skipif(
    not _HAS_GITLEAKS,
    reason=(
        "gitleaks is not installed. The export gate requires it and refuses "
        "without it, so a PASS could not be distinguished from that refusal."
    ),
)

#: File suffixes the gate's ``grep -rI`` would read as text.
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


# ---------------------------------------------------------------------------
# Probes built at run time, never written down
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _script_array(name: str) -> tuple[str, ...]:
    """The entries of a one-per-line bash array in the export script.

    Parsed rather than restated, so a list this file checks cannot go on being
    checked after the script stopped using it. Deliberately duplicated rather
    than imported from a sibling test module: this one has to keep working
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


@functools.lru_cache(maxsize=None)
def _checkout_directory_probe() -> str:
    """The private checkout's directory name in PATH form, derived, never pasted.

    The library lives in a subtree of the development monorepo, so the monorepo
    checkout is this file's grandparent. The trailing slash is the whole point:
    it is what separates a directory from a product name that happens to share
    the token.
    """
    toplevel = subprocess.run(
        ["git", "-C", str(LIB_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert toplevel.returncode == 0, (
        "could not ask git for the checkout root, so the probe could not be "
        f"built. This is could-not-check, not a pass: {toplevel.stderr.strip()}"
    )
    root = Path(toplevel.stdout.strip()).resolve()
    assert root == REPO_ROOT, (
        f"the git checkout root is {root}, not {REPO_ROOT}. The probe assumes "
        "the library is a subtree of the checkout it is exported from; with that "
        "assumption broken it would test the wrong string."
    )
    assert root.name and root.name not in {"/", "."}, f"implausible checkout name {root.name!r}"
    return f"{root.name}/"


@functools.lru_cache(maxsize=None)
def _excluded_deck_name() -> str:
    """The presentation ``EXCLUDES`` drops, read from the list rather than typed."""
    decks = [e for e in _script_array("EXCLUDES") if e.endswith(".pptx")]
    assert len(decks) == 1, f"expected exactly one excluded .pptx entry, found {decks}"
    return PurePosixPath(decks[0]).name


def _is_dropped(rel: str) -> bool:
    """Would the export delete this subtree-relative path?"""
    for entry in _script_array("EXCLUDES"):
        trimmed = entry.rstrip("/")
        if rel == trimmed or rel.startswith(trimmed + "/"):
            return True
    name = PurePosixPath(rel).name
    return any(fnmatch.fnmatch(name, glob) for glob in _script_array("EXCLUDE_GLOBS"))


@functools.lru_cache(maxsize=None)
def _shipped_text_files() -> tuple[str, ...]:
    """Every tracked, non-excluded, text-shaped path the export would publish."""
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", LIB_ROOT.name],
        capture_output=True,
        check=False,
    )
    assert listing.returncode == 0, (
        "git ls-files failed, so the shipped file set could not be enumerated. "
        f"That is could-not-check: {listing.stderr.decode(errors='replace').strip()}"
    )
    prefix = LIB_ROOT.name + "/"
    out = []
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
    assert len(out) > 300, f"only {len(out)} shipped text files found; the enumeration is wrong"
    return tuple(sorted(out))


def _run_gate(directory: Path) -> subprocess.CompletedProcess[str]:
    """Execute the REAL leak gate over ``directory``, read only, never a copy of it."""
    return subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--gate-only", str(directory)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# 1. The private checkout directory name
# ---------------------------------------------------------------------------


@monorepo_only
def test_the_recon_denylist_carries_the_checkout_directory_in_path_form() -> None:
    """The refusal pin. Without this entry nothing looks for the directory form."""
    probe = _checkout_directory_probe()
    patterns = _script_array("LEAK_PATTERNS_RECON")
    assert probe in patterns, (
        f"no entry in LEAK_PATTERNS_RECON matches this checkout's own directory "
        f"name in path form. The slug entry does not cover it, so the private "
        f"checkout can reach the public tree through any usage line that names a "
        f"path inside it. Patterns: {patterns}"
    )


@monorepo_only
def test_no_shipped_file_names_the_private_checkout_directory() -> None:
    """The other half. A pattern with a live match aborts every export."""
    probe = _checkout_directory_probe()
    needle = re.compile(re.escape(probe), re.IGNORECASE)
    offenders = []
    for rel in _shipped_text_files():
        text = (LIB_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if needle.search(line):
                offenders.append(f"{rel}:{number}")
    assert not offenders, (
        "these files would be published naming the private development "
        f"checkout's own directory: {offenders}. Name a path relative to the "
        "repository root instead, which is what a reader of the public "
        "repository actually has."
    )


@monorepo_only
def test_the_gate_refuses_a_planted_checkout_directory_path(tmp_path: Path) -> None:
    """Sabotage of the source, executed against the real gate.

    A denylist entry that is never exercised proves nothing. This plants the
    exact form the docstring used to carry and requires the gate to refuse it.
    """
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "usage_note.md").write_text(
        f"Run it with the {_checkout_directory_probe()}.venv interpreter.\n",
        encoding="utf-8",
    )
    result = _run_gate(planted)
    assert result.returncode != 0, (
        "the gate ACCEPTED a tree naming the private checkout directory:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "recon pattern" in result.stderr and "usage_note.md" in result.stderr, (
        "the gate exited non-zero, but not because of the planted token, so this "
        "test would pass for the wrong reason (a missing scanner refuses too):\n"
        f"{result.stdout}\n{result.stderr}"
    )


@monorepo_only
@_NEEDS_GITLEAKS
def test_the_gate_still_accepts_the_product_name_without_a_path(tmp_path: Path) -> None:
    """Over-correction control, with the measured verdict asserted.

    Widening the entry to the bare token would abort every export on the
    published marketing asset whose filename carries it. The trailing slash is
    load bearing, so a tree holding the token WITHOUT one must still pass.
    """
    bare = _checkout_directory_probe().rstrip("/")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "business_note.md").write_text(
        f"Watch the explainer: https://media.example.org/vfairness_Decoding_{bare}.mp4\n",
        encoding="utf-8",
    )
    result = _run_gate(clean)
    assert result.returncode == 0, (
        "the gate REFUSED a tree whose only match is the product name with no "
        "path separator. That is the over-correction this entry is shaped to "
        f"avoid:\n{result.stdout}\n{result.stderr}"
    )
    assert "Leak gate PASSED" in result.stdout, (
        f"the gate exited 0 without reporting a pass:\n{result.stdout}\n{result.stderr}"
    )


def test_the_gallery_script_documents_a_command_a_public_reader_can_run() -> None:
    """The source fix itself, checked where it is read: the module docstring.

    Runs in the public repository too, because that is where the command has to
    work.
    """
    source = GALLERY_SCRIPT.read_text(encoding="utf-8")
    docstring = source.split('"""')[1]
    command = "PYTHONPATH=src .venv/bin/python scripts/regenerate_gallery_examples.py"
    assert command in docstring, (
        "the documented command no longer names an interpreter path relative to "
        f"the repository root. Docstring:\n{docstring}"
    )
    assert ".." not in docstring, (
        "the documented command reaches outside the repository, which in the "
        "public repository points at nothing and in the private one points at "
        f"the checkout the gate refuses. Docstring:\n{docstring}"
    )
    # Over-correction control: the usage note still says what it said, with
    # measured content rather than a shorter sentence that says less.
    for required in ("numpy", "pandas", "scipy", "sklearn", "jinja2"):
        assert required in docstring, f"the environment note lost {required!r}"
    notebook = re.search(r"notebooks/[A-Za-z0-9_]+\.ipynb", docstring)
    assert notebook, f"the usage note no longer names the notebook it mirrors:\n{docstring}"
    assert (LIB_ROOT / notebook.group(0)).is_file(), (
        f"the docstring names {notebook.group(0)}, which is not in this checkout"
    )


# ---------------------------------------------------------------------------
# 2. Shipped references to paths the export drops
# ---------------------------------------------------------------------------

#: Files that still name a path ``EXCLUDES`` removes, with the count measured on
#: 2026-09-10 and the reason it is tolerated. Two are OPEN DEFECTS of exactly the
#: kind this section is about and are recorded, not excused:
#:
#:   scripts/generate_universal_bias_corpus.py  its closing line prints a "Run
#:       with:" command naming the internal recall harness, which the export
#:       drops. Reproduced by running the shipped script inside the emitted
#:       tree: it writes nine CSVs and then names a file that is not there.
#:   docs/RELEASE_PIPELINE.md  step 3 of "Cutting a release" says to run the
#:       release-tense script, which the export drops on purpose because the cut
#:       happens in the development repository. The step is real; the command is
#:       not runnable where the document is read.
#:
#: The rest are deliberate and already explained where they sit: a test that
#: asserts a path must NOT be published, the admission manifest that records the
#: same, the usage-map page that degrades on its own, and comments naming an
#: internal tool as the origin of a rule.
_NAMES_A_DROPPED_PATH: dict[str, int] = {
    # ANNOTATED ON PURPOSE, not an oversight. The release pipeline documents
    # what the maintainers run, and cut_release.py is monorepo tooling that
    # the export drops. The reference stays because the STEP is real and the
    # published wording it produces is what a reader sees; a blockquote beside
    # it says the script is not in the published repository and why. That is
    # the second remedy this test's own message offers, so the entry is kept
    # rather than the sentence gutted.
    "docs/RELEASE_PIPELINE.md": 1,
    "docs/site/api-reference/index.html": 1,
    "docs/site/js/usage-map.js": 1,
    "docs/site/usage-map/index.html": 2,
    "scripts/pulse_universality_sweep.py": 1,
    "src/vfairness/operations/experimentation/analysis.py": 1,
    # ANNOTATED ON PURPOSE. The docstring of the selection-rate test names
    # the recall harness as the place a fabricated 0.0 turned into a false
    # finding, which is what made that defect worth pinning at all. The
    # harness is internal and the export drops it, so the docstring now says
    # in the same paragraph that the file is not published and that the
    # pinned behaviour is documented in docs/API_REFERENCE.md. Gutting the
    # sentence would remove the reason the test exists.
    "tests/test_bgl4_evaluation_2.py": 1,
    "tests/fixtures/export_published_paths.txt": 1,
    "tests/test_no_aggregator_fabricates_a_verdict.py": 2,
    "tests/test_readiness_export.py": 3,
    "tests/test_readiness_pypi.py": 2,
}

_PATH_REFERENCE = re.compile(
    r"(?:scripts|docs|tests|notebooks|src|spikes)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6}"
)


@functools.lru_cache(maxsize=None)
def _republished_globs() -> tuple[str, ...]:
    """Paths the export deletes and then writes back as a cleaned copy.

    Withheld grading waves sit in EXCLUDES and are republished by
    scripts/publish_grading_waves.py (2026-10-01), so a reference to one DOES
    resolve in the public repository. Parsed from the script's own `case` line,
    like EXCLUDES itself, so this cannot drift from what the export does.
    """
    src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    globs = tuple(re.findall(r'case "\$path" in ([^)\s]+)\) waves\+=', src))
    assert globs, f"could not find the republished-waves case line in {EXPORT_SCRIPT}"
    return globs


def _references_to_dropped_paths() -> dict[str, list[str]]:
    """``{shipped file: ["line: dropped path", ...]}`` for paths that exist here."""
    found: dict[str, list[str]] = {}
    for rel in _shipped_text_files():
        text = (LIB_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            for target in sorted(set(_PATH_REFERENCE.findall(line))):
                if target.startswith("docs/audits/"):
                    continue  # its own section below
                if not (LIB_ROOT / target).exists():
                    continue  # never existed here: a different defect
                if not _is_dropped(target):
                    continue
                if any(fnmatch.fnmatch(target, g) for g in _republished_globs()):
                    continue  # deleted, then published as the cleaned copy
                found.setdefault(rel, []).append(f"{number}: {target}")
    return found


@monorepo_only
def test_no_new_shipped_file_points_at_a_path_the_export_drops() -> None:
    """The refusal pin, as a ratchet over a measured baseline."""
    found = _references_to_dropped_paths()
    new = sorted(set(found) - set(_NAMES_A_DROPPED_PATH))
    assert not new, (
        "these files are published naming a path the export deletes, so the "
        f"reference cannot resolve in the public repository: {new}. Make the "
        "referenced path ship, or say in the referring file what is missing and "
        "why. A pointer into a file nobody can open tells a reader nothing."
    )
    grown = {
        rel: (len(sites), _NAMES_A_DROPPED_PATH[rel])
        for rel, sites in found.items()
        if len(sites) > _NAMES_A_DROPPED_PATH[rel]
    }
    assert not grown, f"new dropped-path references (actual, recorded): {grown}"


@monorepo_only
def test_the_two_documentation_lists_no_longer_link_the_unpublished_deck() -> None:
    """The sites this change closed, plus the control that they were not gutted."""
    deck = _excluded_deck_name()
    for rel in ("docs/API_REFERENCE.md", "docs/LIBRARY_OVERVIEW.md"):
        text = (LIB_ROOT / rel).read_text(encoding="utf-8")
        assert deck not in text, (
            f"{rel} links {deck}, which EXCLUDES removes, so the link is a "
            "guaranteed 404 in the public repository."
        )
    # Over-correction control, measured 2026-09-10: the lists are still lists.
    # Deleting the section would satisfy the assertion above and lose the reader
    # every pointer that does work.
    for rel, heading, expected in (
        ("docs/API_REFERENCE.md", "## Need Help?", 5),
        ("docs/LIBRARY_OVERVIEW.md", "## Resources", 8),
    ):
        text = (LIB_ROOT / rel).read_text(encoding="utf-8")
        assert heading in text, f"{rel} lost the {heading!r} section entirely"
        body = text.split(heading, 1)[1].split("\n---", 1)[0]
        bullets = [line for line in body.splitlines() if line.startswith("- **")]
        assert len(bullets) == expected, (
            f"{rel} {heading!r} has {len(bullets)} entries, expected {expected}. "
            "Removing a dead link is the fix; removing the living ones is not."
        )


# ---------------------------------------------------------------------------
# 3. The internal audit inventory
# ---------------------------------------------------------------------------

#: Files that still name an internal audit document, with the count measured on
#: 2026-09-10. Every one is outside this change's ownership; each is a test or a
#: page whose docstring cites the audit its assertions came from. Recorded so the
#: list can shrink and cannot grow.
_NAMES_AN_AUDIT_DOCUMENT: dict[str, int] = {
    # A REAL CODE PATH, not a dangling citation. That module tests the
    # verdict-ledger builder, so it has to name the ledger it reads, and it
    # guards every such test with an explicit skip when the builder or the
    # ledger is absent (which is always, in the public repository). The
    # thing this ratchet exists to stop is PROSE that points a reader at a
    # document they cannot open; a guarded path that degrades to a skip
    # tells them nothing and denies them nothing.
    "tests/test_readiness3_recollapse.py": 1,
    "docs/QUALITY_AND_HARDENING.md": 1,
    "src/vfairness/validity/__init__.py": 1,
    "tests/test_audit3_eval_core_stats.py": 1,
    "tests/test_audit3_vision_render_tail.py": 1,
    "tests/test_audit_wave6_cicd_nan.py": 1,
    "tests/test_audit_wave6_drift.py": 1,
    "tests/test_audit_wave6_gate_report.py": 1,
    "tests/test_audit_wave6_metrics.py": 1,
    "tests/test_audit_wave6_regression_controls.py": 1,
    "tests/test_docs_truth.py": 1,
    "tests/test_gallery_artwork_matches_code.py": 1,
    "tests/test_no_silent_swallow_core.py": 1,
    "tests/test_packaging_hygiene.py": 1,
    "tests/test_register_pins_resolve.py": 3,
    "tests/test_rowlevel_robustness_training.py": 1,
    "tests/test_site_links_truth.py": 1,
}

#: The files this change cleaned. Named here so a reappearance is a failure with
#: a name on it rather than a silent entry in the ratchet above.
_CLEANED_OF_AUDIT_NAMES = (
    "CHANGELOG.md",
    "docs/API_REFERENCE.md",
    "docs/DIVERGENCES.md",
    "docs/LIBRARY_OVERVIEW.md",
    "docs/ROADMAP.md",
)


@functools.lru_cache(maxsize=None)
def _audit_document_stems() -> tuple[str, ...]:
    """The internal record names, read from the directory rather than typed here."""
    stems = sorted({p.stem for p in AUDIT_DIR.iterdir() if p.is_file()})
    assert stems, f"{AUDIT_DIR} holds no documents, so no probe could be built"
    return tuple(stems)


def _files_naming_an_audit_document() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in _shipped_text_files():
        text = (LIB_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        total = sum(text.count(stem) for stem in _audit_document_stems())
        if total:
            counts[rel] = total
    return counts


@monorepo_only
@audits_present
def test_the_files_this_change_owns_name_no_internal_audit_document() -> None:
    """The refusal pin for the sites that were fixed."""
    counts = _files_naming_an_audit_document()
    regressed = {rel: counts[rel] for rel in _CLEANED_OF_AUDIT_NAMES if rel in counts}
    assert not regressed, (
        "these files name an internal audit document again, by a filename a "
        f"public reader cannot open: {regressed}. Keep the finding, the count "
        "and the date; drop the filename."
    )


@monorepo_only
@audits_present
def test_no_new_shipped_file_names_an_internal_audit_document() -> None:
    """And the ratchet over what is still open elsewhere."""
    counts = _files_naming_an_audit_document()
    new = sorted(set(counts) - set(_NAMES_AN_AUDIT_DOCUMENT))
    assert not new, (
        f"these published files name an internal audit document: {new}. The "
        "document is on EXCLUDES, so the reader is told an assessment exists and "
        "denied its content. State the finding instead of citing the file."
    )
    grown = {
        rel: (count, _NAMES_AN_AUDIT_DOCUMENT[rel])
        for rel, count in counts.items()
        if count > _NAMES_AN_AUDIT_DOCUMENT[rel]
    }
    assert not grown, f"new audit-document citations (actual, recorded): {grown}"


def test_dropping_the_filenames_kept_what_the_references_carried() -> None:
    """Over-correction control, asserting the MEASURED content that had to survive.

    Deleting each sentence would satisfy every assertion above. These are the
    facts those sentences carried: the dates, the counts, the decision and the
    pointers that resolve. Runs in the public repository too, because that is
    where the substitute has to hold up.
    """
    expected: dict[str, tuple[str, ...]] = {
        "CHANGELOG.md": (
            "release candidate on 2026-08-27",
            "84 code findings",
            "71 of them adversarially confirmed",
            "124\n  documentation findings",
            "A third audit, on 2026-08-22",
            "raised 29 findings",
        ),
        "docs/DIVERGENCES.md": (
            "50 percent relative difference",
            "was executed on 2026-08-22",
        ),
        "docs/ROADMAP.md": (
            "### Implementation Plan",
            "Milestones M0 through M8",
            "licence-clearance register",
            "M0 (judge choice and licence sign-off) is complete",
            "Next is M1 (context",
        ),
        "docs/LIBRARY_OVERVIEW.md": (
            "`docs/ROADMAP.md` (Implementation Plan) for the milestone state",
            "licence-clearance register",
        ),
        "docs/API_REFERENCE.md": (
            "(Implementation Plan) carries the",
            "licence-clearance register",
        ),
    }
    for rel, needles in expected.items():
        text = (LIB_ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, (
                f"{rel} no longer states {needle!r}. The filename was the part to "
                "drop; the finding, the date and the count are the part to keep."
            )
    # The replacement pointers must resolve, or this traded one dead reference
    # for another.
    roadmap = (LIB_ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "### Implementation Plan" in roadmap, (
        "docs/API_REFERENCE.md and docs/LIBRARY_OVERVIEW.md now send the reader "
        "to the roadmap's Implementation Plan, and it is not there."
    )


if __name__ == "__main__":  # pragma: no cover - convenience only
    sys.exit(pytest.main([__file__, "-q"]))
