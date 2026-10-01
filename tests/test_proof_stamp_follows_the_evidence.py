"""The stamped proof block may not claim more than the live evidence establishes.

WHAT WENT WRONG. ``src/vfairness/_proof_status.py`` is generated from a capability census
measured on 2026-09-11, and ``scripts/stamp_proof_status.py`` writes each row into the
capability's docstring, where it reaches a reader through ``help()``, an editor hover and
the published API reference. 213 such blocks exist.

Measured on 2026-09-27, after the BGL3 campaign: **59 of the 215 registered capabilities**
carried ``batch: BGL-A`` with ``open_defects: 0`` while a live defect sat in the unit the
row vouches for. The plainest case is ``decoding_trust_runner``, stamped PROVEN while
three of its eight dimensions were fabricating, one of them reporting perfect parity
across all 24 demographic groups because the decision words were matched as bare
substrings.

NOTHING ENFORCED THE OBVIOUS RULE. A capability with a recorded open defect could carry a
PROVEN stamp indefinitely, because the stamp repeated a figure from a dated census and
never consulted the ledger. That is how 59 rows drifted for sixteen days. This file is
that rule.

AND THE CLAIM WAS WRONG IN BOTH DIRECTIONS. The old BGL-A sentence asserted that "an
independent judge showed" the value to be a true measurement, which is unearned for every
one of the 342 grades this campaign produced: each came from one examiner and none has
been argued with, and roughly one grade in five has been overturned when somebody did.
The same sentence DISCLAIMED that the pin had been sabotage-checked, which understates
evidence that now exists. Understating teaches a reader to discount the whole block,
which costs as much as overstating it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "capability-status.json"


@pytest.fixture(scope="module")
def stamper():
    spec = importlib.util.spec_from_file_location(
        "_stamp_proof_status", ROOT / "scripts" / "stamp_proof_status.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stamp_proof_status"] = module
    spec.loader.exec_module(module)
    return module


class _Fake:
    """Stands in for a capability object, resolved by module and qualname.

    The CLASS is passed to _facts_for_today, never an instance: ``__qualname__`` is a
    descriptor on ``type``, so an instance does not see it, ``_units_of`` returns an
    empty list and every assertion below then passes over nothing. That is exactly the
    shape of check this repository audits for, so it is worth the sentence.
    """

    __module__ = "vfairness.fake"
    __qualname__ = "FakeCapability"


def test_a_unit_in_fix_pending_forces_the_block_off_proven(stamper):
    """THE GATE, exercised on the mechanism rather than on today's data.

    FIX PENDING is empty right now, because this campaign closed every open defect, so an
    assertion over the live ledger would pass while establishing nothing. The rule is
    therefore asked of the function that decides it, with a status map that contains what
    the live one currently does not.
    """
    status = {
        "vfairness.fake.FakeCapability": "CHECKED",
        "vfairness.fake.FakeCapability.good": "CHECKED",
        "vfairness.fake.FakeCapability.bad": "FIX PENDING",
    }
    facts = stamper._facts_for_today(_Fake, status, {})
    assert facts["open_now"] == 1, facts
    # And the healthy shape must NOT trip it, or the gate would force every capability
    # down to BGL-D and the ladder would stop meaning anything.
    clean = {k: "CHECKED" for k in status}
    assert stamper._facts_for_today(_Fake, clean, {})["open_now"] == 0


def test_the_independent_check_is_claimed_only_where_a_grade_is_audited(stamper):
    """`audited` fails closed: one unaudited unit means the capability is not audited."""
    status = {
        "vfairness.fake.FakeCapability.a": "CHECKED",
        "vfairness.fake.FakeCapability.b": "CHECKED",
    }
    both = {
        "vfairness.fake.FakeCapability.a": {"grade": "PROVEN", "audited": True, "sabotage": "x"},
        "vfairness.fake.FakeCapability.b": {"grade": "PROVEN", "audited": True, "sabotage": "x"},
    }
    assert stamper._facts_for_today(_Fake, status, both)["audited"] is True
    one_short = dict(both)
    one_short["vfairness.fake.FakeCapability.b"] = {
        "grade": "PROVEN",
        "audited": False,
        "sabotage": "x",
    }
    assert stamper._facts_for_today(_Fake, status, one_short)["audited"] is False


def test_a_unit_that_is_not_a_measurement_does_not_owe_a_sabotage(stamper):
    """It has nothing to pin, so requiring one fails closed where the question does not
    apply. Measured: the reweighting capability covers four units, one of them NOT A
    MEASUREMENT, and counting it reported "the pin has not been sabotage-checked" for a
    capability whose two measuring units were sabotaged twenty times the same day."""
    status = {
        "vfairness.fake.FakeCapability.measures": "CHECKED",
        "vfairness.fake.FakeCapability.does_not": "CHECKED",
    }
    grades = {
        "vfairness.fake.FakeCapability.measures": {
            "grade": "PROVEN",
            "audited": False,
            "sabotage": "re-broken, went red",
        },
        "vfairness.fake.FakeCapability.does_not": {
            "grade": "NOT A MEASUREMENT",
            "audited": True,
            "sabotage": "",
        },
    }
    assert stamper._facts_for_today(_Fake, status, grades)["sabotaged"] is True
    # But a MEASURING unit with no sabotage record still withholds the claim.
    grades["vfairness.fake.FakeCapability.measures"]["sabotage"] = ""
    assert stamper._facts_for_today(_Fake, status, grades)["sabotaged"] is False


def _blocks(text: str) -> list[str]:
    """Each proof block with its line wrapping removed.

    textwrap.fill breaks the block at 88 columns, so "NOT INDEPENDENTLY CHECKED" can
    arrive as "NOT INDEPENDENTLY\nCHECKED" and a plain substring search for the phrase
    finds nothing. The first version of the test below searched the raw text and reported
    that NO block disclosed an unaudited grade while every one of the 213 did.
    """
    out = []
    for raw in re.findall(
        r"Beta Go-Live proof status \(.*?\(end Beta Go-Live proof status\)", text, re.S
    ):
        out.append(" ".join(raw.split()))
    return out


def test_no_stamped_block_claims_an_independent_check_that_did_not_happen():
    """Over the LIVE tree, and it is the flattering direction that matters.

    Not one grade this campaign produced is audited, so no block may assert that an
    independent check upheld it, and every one of them must say so instead.
    """
    src = ROOT / "src" / "vfairness"
    claimed, disclosed, total = [], 0, 0
    for path in src.rglob("*.py"):
        for block in _blocks(path.read_text(encoding="utf-8")):
            total += 1
            if "An independent check has argued with that judgement" in block:
                claimed.append(str(path.relative_to(src)))
            if "NOT INDEPENDENTLY CHECKED" in block:
                disclosed += 1
    assert total > 100, f"only {total} proof blocks were parsed; the block shape changed"
    assert disclosed, (
        f"none of {total} blocks discloses an unaudited grade, yet the release gate "
        f"reports 556 grades with no independent check. The stamp has stopped "
        f"following the evidence."
    )
    audited_exists = any(
        g.get("audited")
        for g in _graded_now().values()
        if g.get("grade") and g.get("grade") != "NOT A MEASUREMENT"
    )
    if not audited_exists:
        assert not claimed, (
            f"these blocks assert an independent check while NOTHING on the surface is "
            f"audited: {sorted(set(claimed))[:6]}"
        )


def _graded_now() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import library_kpis

    return dict(library_kpis.graded_items())


def test_every_stamp_is_dated_to_the_evidence_not_to_the_census():
    """The census ran on 2026-09-11 and these blocks now state what the live ledger
    establishes. A newer claim under an older date is a stale figure dressed as a
    measurement, which is the shape this whole file is about."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    newest = sorted(d for d in (ledger.get("evidence_dates") or []) if d)[-1]
    src = ROOT / "src" / "vfairness"
    wrong = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "Beta Go-Live proof status (" in line and f"({newest})" not in line:
                wrong.append(f"{path.relative_to(src)}: {line.strip()[:70]}")
    assert not wrong, (
        f"these blocks are dated other than {newest}, the newest evidence date:\n  "
        + "\n  ".join(wrong[:8])
    )
