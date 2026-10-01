"""The published second-round-audit count must match the audit files.

The register in ``docs/bgl6-audit-register.json`` and the blocks it fills in
``docs/QUALITY_AND_HARDENING.md`` and the rendered quality page are the only place a
reader can see how much of the second audit is closed. A number a reader can see and
nobody checks is the defect this repository is audited for, so it is checked here.

WHY THE STATE IS DECLARED. A test that RECORDS a defect asserts the value the unit
produces today, so it PASSES while the defect is live; a pin asserts the corrected
value and also passes. Execution cannot tell them apart, so each audit file declares
``RECORDED_DEFECTS_STILL_OPEN`` and this file refuses a name in that list which is not
a test function in the module. The declaration shrinks in the same commit that fixes a
defect and inverts its test.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "bgl6-audit-register.json"
AUDIT_FILES = sorted((ROOT / "tests").glob("test_bgl6_f*.py"))


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_audit_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _test_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def test_there_are_audit_files_to_count():
    """Without this the whole file passes over an empty sweep."""
    assert len(AUDIT_FILES) >= 6, [p.name for p in AUDIT_FILES]


@pytest.mark.parametrize("path", AUDIT_FILES, ids=lambda p: p.stem)
def test_every_audit_file_declares_what_is_still_open(path: Path):
    """A file with no declaration would silently count as fully closed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # ast.AnnAssign as well as ast.Assign: a batch with nothing left open declares
    # ``RECORDED_DEFECTS_STILL_OPEN: list[str] = []``, and an Assign-only reader
    # calls that file undeclared.
    declared = [
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    ] + [
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert "RECORDED_DEFECTS_STILL_OPEN" in declared, (
        f"{path.name} declares no RECORDED_DEFECTS_STILL_OPEN, so the register would "
        "read it as having nothing open"
    )


@pytest.mark.parametrize("path", AUDIT_FILES, ids=lambda p: p.stem)
def test_no_open_name_is_a_test_that_does_not_exist(path: Path):
    """A stale name would hold the open count up after its test was renamed away,
    and a typo would hold it up forever. Either way the published number would be
    about something that is not there."""
    names = set(_test_names(path))
    declared = list(_module(path).RECORDED_DEFECTS_STILL_OPEN)
    unknown = [n for n in declared if n not in names]
    assert not unknown, f"{path.name} lists tests that do not exist: {unknown}"
    assert len(set(declared)) == len(declared), f"{path.name} lists a name twice"


def test_the_published_register_matches_the_audit_files():
    """The ratchet. Runs the generator's own --check, which also verifies the block
    on the rendered quality page, so the markdown and the page cannot drift apart."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bgl6_register.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "the published second-round-audit count disagrees with the audit files. Run "
        "`python scripts/bgl6_register.py`.\n" + proc.stdout + proc.stderr
    )


def test_the_totals_add_up_and_are_not_vacuous():
    """A register of zero claims would satisfy every check above."""
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    totals = reg["totals"]
    assert totals["recorded"] >= 50, totals
    assert totals["closed"] + totals["open"] == totals["recorded"], totals
    by_batch = sum(b["recorded"] for b in reg["batches"].values())
    assert by_batch == totals["recorded"], (by_batch, totals)
    assert sum(b["open"] for b in reg["batches"].values()) == totals["open"]
    assert sum(b["closed"] for b in reg["batches"].values()) == totals["closed"]


def test_the_markdown_block_states_the_same_totals():
    """The rendered page is checked by the generator's --check above; this is the
    markdown twin, asserted separately because they are different files."""
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    t = reg["totals"]
    text = (ROOT / "docs" / "QUALITY_AND_HARDENING.md").read_text(encoding="utf-8")
    assert "<!-- BGL:second_round:start -->" in text
    assert f"**{t['recorded']} claims**" in text, t
    assert f"**{t['closed']} are closed**" in text, t
    assert f"**{t['open']} are still open**" in text, t


def test_the_beta_gate_reports_the_registers_own_numbers():
    """B2b must MEASURE the register, not merely mention it.

    Found by sabotage on 2026-09-28: blinding the gate's register reader so it
    returned zeros made B2b report "0 open of 0 recorded claims" and PASS, and every
    test in tests/test_release_gate.py stayed green. A criterion that can be switched
    off without anything going red is a criterion that does not gate, which is the
    defect this whole register exists to make visible, sitting in the gate itself.

    So the two numbers are compared here. If B2b ever stops reading the register, the
    register's own honesty file is where it shows up.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["totals"]
    b2b = next(c for c in release_gate._beta_criteria() if c["id"] == "B2b")

    assert f"{reg['open']} open" in b2b["measured"], (
        f"B2b reports {b2b['measured']!r} while the register says {reg['open']} open"
    )
    assert f"{reg['recorded']} recorded" in b2b["measured"], (
        f"B2b reports {b2b['measured']!r} while the register records {reg['recorded']}"
    )
    assert b2b["passes"] is (reg["open"] == 0), (
        f"B2b passes={b2b['passes']} with {reg['open']} open records"
    )
    # AND THE REGISTER IS NOT EMPTY, so the equality above cannot be satisfied by two
    # zeros. This is the exact state the sabotage produced.
    assert reg["recorded"] >= 50, reg


def test_every_beta_criterion_carries_a_reader_facing_sentence():
    """The status board draws one coloured block per criterion, and a block with no
    explanation is a colour with no meaning.

    Daniel asked what the B sections are after looking at the board, which is the
    right question to have to ask once and the wrong one to have to ask twice. The
    sentence lives on the criterion in scripts/release_gate.py, so the generated
    fallback and the page's own runtime rebuild read ONE source; this refuses a
    criterion that reaches the board without one.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    missing = [c["id"] for c in release_gate._beta_criteria() if not (c.get("plain") or "").strip()]
    assert not missing, (
        f"{missing} would be drawn on the board as an unexplained coloured block. Add a "
        "sentence to PLAIN_ENGLISH in scripts/release_gate.py."
    )


def test_the_board_renders_an_explanation_for_every_criterion():
    """And the sentence has to reach the page, not just exist.

    The board is generated, so this reads the rendered file: one tooltip per
    criterion, each carrying its id and its measured value, so the no-JavaScript
    view explains itself as well as the live one.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    page = (ROOT / "docs" / "site" / "quality-and-hardening" / "index.html").read_text(
        encoding="utf-8"
    )
    board = page[
        page.index("<!-- BGL6:readiness:start -->") : page.index("<!-- BGL6:readiness:end -->")
    ]
    beta = release_gate._beta_criteria()
    tips = board.count('class="qa-seg__tip"')
    assert tips == len(beta), f"{tips} tooltips for {len(beta)} criteria"
    for c in beta:
        assert c["plain"][:40] in board, f"{c['id']}'s explanation is not on the board"
        assert str(c.get("measured", ""))[:24] in board, f"{c['id']} shows no measured value"


def test_the_beta_criterion_table_lists_every_criterion_with_the_gates_own_state():
    """The typed copy of the board's facts, which rotted in the worst direction.

    Measured 2026-09-29, before this guard existed: the table listed FIVE criteria
    while the gate had seven, gave B1 as "1564 of 1564", and reported B2 as a FAIL
    with "64 open (37 measuring, 27 unclassified)" long after B2 reached zero open
    and started passing. A stale Pass is bad. A stale Fail, on the page whose whole
    subject is not overstating what has been checked, is worse: it teaches a reader
    that the numbers here are decorative.

    It is generated from scripts/release_gate.py now, so this test is what keeps it
    that way. It asserts the STATE column, not only the row count, because a table
    can carry every criterion and still say Pass where the gate says Blocks.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import release_gate

    page = (ROOT / "docs" / "site" / "quality-and-hardening" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "<!-- BGL6:plan:start -->" in page and "<!-- BGL6:plan:end -->" in page, (
        "the beta-criterion table is not inside its marker pair, so nothing regenerates it"
    )
    block = page[page.index("<!-- BGL6:plan:start -->") : page.index("<!-- BGL6:plan:end -->")]
    beta = release_gate._beta_criteria()
    for c in beta:
        row_id = 'id="plan-{}"'.format(c["id"].lower())
        assert row_id in block, f"{c['id']} has no row in the beta-criterion table"
        row = block[block.index(row_id) : block.index("</tr>", block.index(row_id))]
        assert str(c.get("measured", ""))[:24] in row, f"{c['id']}'s row shows a stale measurement"
        expected = "Pass" if c["passes"] else "Blocks"
        assert expected in row, f"{c['id']} is {expected} at the gate and the table disagrees"
    rows = block.count('id="plan-')
    assert rows == len(beta), f"{rows} rows for {len(beta)} criteria"


def test_the_beta_criterion_table_is_rebuilt_from_the_gate_at_runtime():
    """Generated is not enough on its own: the page is served long after it is built.

    The board is rebuilt on load from library-stats.json, and this table has to be
    rebuilt from the SAME reading, or the two disagree for every reader who opens
    the page between one regeneration and the next.
    """
    page = (ROOT / "docs" / "site" / "quality-and-hardening" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "querySelector('.kpi-plan-body')" in page, (
        "nothing rebuilds the beta-criterion table on load"
    )
    rebuild = page[page.index("querySelector('.kpi-plan-body')") :][:900]
    assert "crit" in rebuild, "the rebuild does not read the gate's criteria"
    assert "qa-state--fail" in rebuild, "the rebuild cannot render a blocking state"
