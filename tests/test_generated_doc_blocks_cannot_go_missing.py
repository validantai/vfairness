"""A generated block with one marker is refused, not skipped.

WHY. scripts/beta_go_live_docs.py renders the release figures into the markdown
between `<!-- BGL:name:start -->` and `<!-- BGL:name:end -->`, and `--check` fails
the build when a rendered block is stale. Its loop opened with

    if begin not in out:
        continue

so DELETING A START MARKER silently disabled that block, and --check answered
"docs are current" over whatever prose had replaced the generated table.

Found 2026-09-27 by sabotaging the new gate block: the generated beta and full
release verdicts were replaced by the single hand-typed sentence "Four of the six
beta criteria pass", in the paragraph a reader consults to decide whether the
library can be published, and nothing noticed. That sentence had in fact been true
two days earlier and was false when the sabotage was applied.

A gate that cannot notice its own subject going missing is the defect this
repository is audited for, committed in the generator. Half a marker pair is now an
error that names the missing marker.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Kept in step with scripts/beta_go_live_docs.py's own render list, and pinned
# below rather than trusted: a target added there and forgotten here would have its
# markers unchecked, which is the same hole as the one this file was written for.
TARGETS = (
    ROOT / "docs" / "BETA_GO_LIVE_PLAN.md",
    ROOT / "docs" / "QUALITY_AND_HARDENING.md",
    ROOT / "docs" / "RELEASE_PLAN.md",
    ROOT / "docs" / "BETA.md",
)


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_beta_go_live_docs_under_test", ROOT / "scripts" / "beta_go_live_docs.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def bgl():
    return _mod()


def _blocks_in(text: str, names) -> list[str]:
    return [n for n in names if f"<!-- BGL:{n}:start -->" in text]


def test_the_targets_carry_blocks_at_all(bgl):
    """Guards the guard: with no blocks anywhere the rest is vacuous."""
    found = {
        t.name: _blocks_in(t.read_text(encoding="utf-8"), bgl.BLOCKS) for t in TARGETS if t.exists()
    }
    assert found, "none of the target documents exists"
    assert sum(len(v) for v in found.values()) >= 5, f"too few generated blocks: {found}"


def test_every_marker_is_paired_in_every_target(bgl):
    unpaired = []
    for target in TARGETS:
        if not target.exists():
            continue
        text = target.read_text(encoding="utf-8")
        for name in bgl.BLOCKS:
            b, e = f"<!-- BGL:{name}:start -->", f"<!-- BGL:{name}:end -->"
            if (b in text) != (e in text):
                unpaired.append(f"{target.name}: {name} has one marker, not both")
    assert not unpaired, "\n".join(unpaired)


@pytest.mark.parametrize("which", ["start", "end"])
def test_removing_either_marker_makes_the_checker_refuse(bgl, tmp_path, which):
    """The mechanism, exercised. Either half missing must be an error.

    Run on a COPY so the repository's own documents are never modified by a test.
    """
    src = ROOT / "docs" / "QUALITY_AND_HARDENING.md"
    names = _blocks_in(src.read_text(encoding="utf-8"), bgl.BLOCKS)
    assert names, "no generated block in the document under test"
    name = names[0]

    copy = tmp_path / src.name
    shutil.copyfile(src, copy)
    text = copy.read_text(encoding="utf-8")
    copy.write_text(text.replace(f"<!-- BGL:{name}:{which} -->", "", 1), encoding="utf-8")

    assert bgl.render(copy, check=True) is False, (
        f"removing the {which} marker of block {name!r} left the checker satisfied, so "
        "the block's content is neither generated nor checked"
    )


def test_the_control_an_intact_copy_is_accepted(bgl, tmp_path):
    """Over-correction control. Refusing every document would pass the test above."""
    src = ROOT / "docs" / "QUALITY_AND_HARDENING.md"
    copy = tmp_path / src.name
    shutil.copyfile(src, copy)
    assert bgl.render(copy, check=True) is True, (
        "an unmodified copy of the document was refused, so the marker rule is "
        "rejecting valid files"
    )


def test_this_file_checks_every_document_the_generator_renders():
    """A target the generator renders and this file does not name goes unchecked.

    The generator's render list lives in its main() and grew from two documents to
    four in one evening. Reading it here rather than restating it means a fifth
    target cannot be added on one side only.
    """
    import re

    src = (ROOT / "scripts" / "beta_go_live_docs.py").read_text(encoding="utf-8")
    rendered = set(re.findall(r'"(docs/[A-Za-z0-9_]+\.md)"', src))
    assert rendered, "could not read the generator's render list; the regex has drifted"
    named = {str(t.relative_to(ROOT)) for t in TARGETS}
    missing = sorted(rendered - named)
    assert not missing, (
        f"the generator renders {missing} and this file does not check them, so their "
        "markers are unchecked"
    )
