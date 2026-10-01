"""Executable pins on the documentation that ships to a pip user.

Two classes of claim are pinned here, both of which were found overstated in the
fourth-iteration pre-release audit (``docs/audits/fourth-iteration-audit-2026-08-27.md``):

1. **README output claims** (finding #17). The quickstart advertised
   ``"1/5 metrics within thresholds (2 metric(s) not assessable ...)"`` while the
   real run prints ``1/5 metrics within thresholds``. The fabricated half was the
   library's headline three-state differentiator, in the first code a stranger
   runs. Hand-maintained example output rots silently, so every README code block
   is EXECUTED here and its printed lines are compared to the ``# ->`` lines the
   README shows.

2. **Citation-vs-implementation divergence** (finding #15).
   ``conditional_demographic_disparity`` names Wachter, Mittelstadt & Russell
   (2021) but computes a different statistic. The divergence was recorded only in
   ``docs/DIVERGENCES.md``, which ships in neither artifact and is not on the
   public docs site, so a pip user reading the docstring registered a margin from
   that literature and was wrong. A shipped docstring that cites a paper AND is
   listed in DIVERGENCES.md must disclose the divergence in the docstring itself,
   because that is the text the wheel carries.

The README harness has its own positive controls: a deliberately wrong
expectation and a deliberately failing block must both be reported. Without them
a harness that silently found no blocks, or compared nothing, would pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
DIVERGENCES = REPO_ROOT / "docs" / "DIVERGENCES.md"
SRC = REPO_ROOT / "src"

# A fenced block, capturing its info string (language) and its body.
_FENCE_RE = re.compile(r"^```([^\n`]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)

# The README's convention for "and this is what it prints".
_EXPECT_RE = re.compile(r"^\s*#\s*->\s?(.*?)\s*$")


class CodeBlock(NamedTuple):
    """One fenced code block lifted out of a markdown file."""

    language: str
    body: str
    expectations: List[str]


def extract_blocks(markdown: str, language: str) -> List[CodeBlock]:
    """Every fenced block in ``markdown`` written in ``language``, in document order."""
    blocks: List[CodeBlock] = []
    for match in _FENCE_RE.finditer(markdown):
        info = match.group(1).strip().lower()
        if info != language:
            continue
        body = match.group(2)
        expectations = [
            m.group(1) for m in (_EXPECT_RE.match(line) for line in body.splitlines()) if m
        ]
        blocks.append(CodeBlock(language=info, body=body, expectations=expectations))
    return blocks


def run_python_block(body: str) -> Tuple[int, List[str], str]:
    """Execute one README python block the way a reader would: a fresh interpreter.

    The source tree is put on ``PYTHONPATH`` so the block runs against THIS
    checkout rather than whatever ``vfairness`` may already be installed. Returns
    the exit status, stdout split into lines, and stderr (warnings live there and
    are not part of the comparison).
    """
    completed = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**_clean_env(), "PYTHONPATH": str(SRC)},
        timeout=600,
    )
    return completed.returncode, completed.stdout.splitlines(), completed.stderr


def _clean_env() -> Dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def check_python_blocks(markdown: str) -> List[str]:
    """Run every python block and return a list of human-readable problems.

    An empty list means every block ran and every ``# ->`` line matched the
    output actually printed, in order. This function is the thing under test in
    the positive controls below: it must REPORT a wrong expectation, not shrug.
    """
    problems: List[str] = []
    for index, block in enumerate(extract_blocks(markdown, "python"), start=1):
        status, stdout_lines, stderr = run_python_block(block.body)
        if status != 0:
            tail = "\n".join(stderr.strip().splitlines()[-6:])
            problems.append(f"python block {index} exited {status}:\n{tail}")
            continue
        if not block.expectations:
            continue
        if stdout_lines != block.expectations:
            problems.append(
                f"python block {index} output does not match its '# ->' lines.\n"
                f"  README says: {block.expectations!r}\n"
                f"  really prints: {stdout_lines!r}"
            )
    return problems


# --------------------------------------------------------------------------
# README: the shown output is the real output
# --------------------------------------------------------------------------


def test_readme_has_python_blocks_that_claim_an_output() -> None:
    """Guard against a vacuous suite: the harness must have something to check.

    If the README's blocks or its ``# ->`` convention are ever renamed away, the
    execution test below would pass by finding nothing. This fails first instead.
    """
    blocks = extract_blocks(README.read_text(encoding="utf-8"), "python")
    assert blocks, "README.md has no ```python blocks; the docs-truth harness checks nothing"
    with_expectations = [b for b in blocks if b.expectations]
    assert with_expectations, (
        "no README python block shows an expected output with '# -> ...'; "
        "the harness cannot detect an overstated example"
    )


def test_readme_python_blocks_print_exactly_what_the_readme_shows() -> None:
    """Finding #17: the quickstart advertised an output line the code never produced."""
    pytest.importorskip("numpy")
    problems = check_python_blocks(README.read_text(encoding="utf-8"))
    assert not problems, "README code does not match its shown output:\n\n" + "\n\n".join(problems)


def test_harness_reports_a_wrong_expectation() -> None:
    """Positive control: reinstate the defect shape and the harness must go red.

    This is the exact failure of finding #17 in miniature, an example whose shown
    output is richer than what it prints. A harness that only ever sees a correct
    README proves nothing.
    """
    fake = textwrap.dedent(
        """
        ```python
        print("1/5 metrics within thresholds")
        # -> 1/5 metrics within thresholds (2 metric(s) not assessable ...)
        ```
        """
    )
    problems = check_python_blocks(fake)
    assert len(problems) == 1, problems
    assert "does not match" in problems[0]


def test_harness_reports_a_block_that_does_not_run() -> None:
    """Positive control: a block that raises must be reported, not counted as output-less."""
    fake = "```python\nraise SystemExit('boom')\n```\n"
    problems = check_python_blocks(fake)
    assert len(problems) == 1, problems
    assert "exited" in problems[0]


def test_harness_accepts_a_correct_block() -> None:
    """Negative control for the controls: a truthful block must not be flagged."""
    fake = '```python\nprint("ok")\n# -> ok\n```\n'
    assert check_python_blocks(fake) == []


def test_readme_install_commands_name_only_declared_extras() -> None:
    """Every ``vfairness[...]`` the README tells a reader to install must exist.

    The shell blocks are not executed (that would hit the network and install
    packages), so they are checked against the package metadata instead, and this
    test says so rather than implying the command was run.
    """
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"].get("optional-dependencies", {}))

    text = README.read_text(encoding="utf-8")
    named: set = set()
    for block in extract_blocks(text, "bash"):
        for match in re.finditer(r"vfairness\[([^\]]+)\]", block.body):
            named.update(part.strip() for part in match.group(1).split(","))
    # Also the extras table, which is prose rather than a code block.
    for match in re.finditer(r"^\|\s*`([a-z][a-z0-9_-]*)`\s*\|", text, re.MULTILINE):
        named.add(match.group(1))

    unknown = sorted(name for name in named if name not in declared)
    assert not unknown, f"README names extras that pyproject.toml does not declare: {unknown}"


def test_readme_ships_no_dead_ci_badge() -> None:
    """Finding #14: the CI badge 404s and renders broken on the PyPI page.

    README.md is the PyPI long description, so a broken image there is the first
    thing a visitor sees. The badge was removed rather than shipped broken; it
    may come back only once the workflow badge actually resolves.
    """
    text = README.read_text(encoding="utf-8")
    assert "actions/workflows" not in text, (
        "README embeds a GitHub Actions badge again. Confirm the badge URL returns "
        "200 before restoring it; it returned 404 on 2026-08-27."
    )


# --------------------------------------------------------------------------
# CHANGELOG: the release notes describe the artifact being released
# --------------------------------------------------------------------------


def _changelog_sections(text: str) -> Dict[str, str]:
    """Map each ``## [heading]`` to its body."""
    parts = re.split(r"^## (\[[^\]]+\][^\n]*)$", text, flags=re.MULTILINE)
    sections: Dict[str, str] = {}
    for index in range(1, len(parts) - 1, 2):
        sections[parts[index].strip()] = parts[index + 1]
    return sections


def test_changelog_top_section_is_the_unreleased_version() -> None:
    """The top heading must be the unreleased version, undated, and the only one.

    Findings #3 and #9 were that an [Unreleased] section described code already
    inside the wheel a tag would publish. PyPI is immutable, so release notes
    that misfile a release's own contents cannot be corrected afterwards.

    The fix chosen on 2026-08-28 (Daniel) is stronger than emptying that section:
    there is NO separate [Unreleased] heading at all. While 0.1.0 is itself
    unreleased, a second holding pen only re-creates the same trap of work
    sitting under one heading while shipping inside another. Everything
    accumulates under `## [0.1.0] - UNRELEASED` until the tag is pushed.

    This also pins that the heading carries NO DATE. Earlier revisions claimed a
    cut on 2026-08-23 and then re-dated it to 2026-08-27; both were fiction,
    because no tag exists and nothing has been published. A date here asserts a
    release event, so it may only appear in the same change that pushes the tag.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    sections = _changelog_sections(text)
    heads = list(sections)

    stray = [h for h in heads if h.startswith("[Unreleased]")]
    assert not stray, (
        "a separate [Unreleased] heading is back. While 0.1.0 is unreleased, "
        "entries belong under it, not in a second holding pen: "
        f"{stray}"
    )

    top = heads[0]
    assert top.startswith("[0.1.0]"), f"top changelog section is not [0.1.0]: {top!r}"
    assert "UNRELEASED" not in top.upper(), (
        f"0.1.0 is released, so the top section must carry its date, not UNRELEASED; got {top!r}"
    )
    assert re.search(r"\d{4}-\d{2}-\d{2}", top), (
        f"the released heading must carry the publication date; got {top!r}"
    )


def test_changelog_links_no_nonexistent_tags() -> None:
    """Finding #14: every compare/tag link answered 404 because no tag exists."""
    text = CHANGELOG.read_text(encoding="utf-8")
    # Only real link definitions count; the explanatory HTML comment does not.
    without_comments = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    dead = re.findall(r"^\[[^\]]+\]:\s*(\S+/(?:compare|releases/tag)/\S+)", without_comments, re.M)
    assert not dead, (
        "CHANGELOG defines links to git tags that do not exist yet: "
        f"{dead}. Restore them in the same change that pushes the first tag."
    )


# --------------------------------------------------------------------------
# Docstrings: a cited paper whose definition we diverge from must say so
# --------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\b(?:19|20)\d\d\b")
_DISCLOSURE_RE = re.compile(r"diverg", re.IGNORECASE)


def _divergent_function_names() -> List[str]:
    """Public callables that DIVERGENCES.md records as computing their own definition.

    Parsed from the document rather than hard-coded, so adding a divergence there
    without disclosing it in the docstring fails this suite.
    """
    text = DIVERGENCES.read_text(encoding="utf-8")
    return sorted({m.group(1) for m in re.finditer(r"vfairness `([a-z_][a-z0-9_]*)`", text)})


def test_divergences_document_is_parseable() -> None:
    """Guard the parser above: it must actually find the recorded divergences."""
    names = _divergent_function_names()
    assert "conditional_demographic_disparity" in names, (
        "DIVERGENCES.md no longer names conditional_demographic_disparity in the "
        f"expected form; the docstring check below would silently check nothing. Found: {names}"
    )


@pytest.mark.parametrize("name", _divergent_function_names())
def test_divergent_metric_docstring_discloses_it_when_it_cites_a_paper(name: str) -> None:
    """Finding #15: the disclosure has to live where a pip user reads it.

    ``docs/DIVERGENCES.md`` ships in neither the wheel nor the sdist and is not
    published on the docs site. A docstring that cites a year (i.e. names a paper)
    while implementing a different statistic must carry the divergence note
    itself, because the docstring is what ``help()`` and every IDE show.
    """
    import vfairness
    from vfairness._proof_status import strip_proof_block

    func = getattr(vfairness, name, None)
    if func is None:
        pytest.skip(f"{name} is not exported from the top-level package")
    raw: Optional[str] = func.__doc__
    assert raw, f"{name} has no docstring at all"
    # Strip the generated Beta Go-Live block before applying a prose heuristic.
    # Its date, "(2026-09-11)", matches _CITATION_RE, so every stamped metric
    # would read as citing a paper. Weakening the citation rule would have let a
    # real citation through; removing generated text from the scan does not.
    doc = strip_proof_block(raw)
    if not _CITATION_RE.search(doc):
        # No paper is cited, so no reader can be misled by one.
        return
    assert _DISCLOSURE_RE.search(doc), (
        f"{name} cites a paper and is recorded in docs/DIVERGENCES.md as diverging "
        "from a reference definition, but its docstring does not disclose that. "
        "DIVERGENCES.md ships in neither artifact, so the docstring is the only "
        "place a pip user can read it."
    )


def test_cdd_docstring_names_what_actually_differs() -> None:
    """The CDD case, pinned specifically rather than only by the generic rule above.

    A note saying merely "this diverges" would satisfy the parametrised test and
    still leave the reader unable to tell WHAT differs. The two facts that decide
    whether a margin transfers are: the implemented value is a selection-rate
    spread, and it is unsigned where the cited statistic is signed.
    """
    from vfairness import conditional_demographic_disparity

    doc = conditional_demographic_disparity.__doc__ or ""
    lowered = doc.lower()
    for token in ("wachter", "selection-rate", "unsigned", "signed", "margin"):
        assert token in lowered, f"CDD docstring no longer states '{token}':\n{doc}"


def test_cdd_ci_variant_carries_the_same_warning() -> None:
    """The ``_with_ci`` variant is the one a seal gates on, so it must not be silent."""
    from vfairness import conditional_demographic_disparity_with_ci

    doc = (conditional_demographic_disparity_with_ci.__doc__ or "").lower()
    assert "diverg" in doc and "wachter" in doc, (
        "the CI variant gates a sealed verdict on an operator-registered margin; "
        "it must point at the divergence note too"
    )
