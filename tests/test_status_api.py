"""`vfairness.status()` must answer about the installed version, and agree with the site.

WHY THE API EXISTS. The three-state status of every capability is published as a
page, which is no use to the two people who need it most: a user who wants their
own pipeline to fail when it depends on something we have not verified, and a user
on an older version, for whom today's page is the wrong answer. So the ledger ships
inside the wheel and `vfairness.status()` reads it.

THE RULE THIS FILE MOSTLY EXISTS TO ENFORCE: the API and the published badge must
never disagree about the same name. They are two renderings of one measurement, and
during development they DID disagree: `status("CalibrationAnalyzer")` reported the
class row alone and said CHECKED, while the badge folded in the twenty methods
reached through it, found one with an open defect, and said FIX PENDING. A class
with a defective method reading CHECKED in code and FIX PENDING on the web is the
exact defect this whole programme exists to remove, one level up.

An UNKNOWN NAME RAISES rather than returning NOT CHECKED. Those are different
answers: a typo must not read as a finding about the library.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import vfairness
from vfairness.status import UnknownCapabilityError

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "capability-status.json"
STATES = ("CHECKED", "FIX PENDING", "NOT CHECKED")


@pytest.fixture(scope="module")
def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def test_the_surface_summary_matches_the_repository_ledger(ledger):
    surface = vfairness.status()
    assert surface.total == len(ledger["units"])
    assert surface.counts == ledger["counts"]
    assert surface.library_version == vfairness.__version__


def test_every_shipped_row_matches_the_repository_ledger(ledger):
    """The shipped copy is the one users read. If it drifts they are told something
    about a state that no longer holds, and nothing in their run would say so."""
    shipped = json.loads(
        (ROOT / "src" / "vfairness" / "_capability_status.json").read_text(encoding="utf-8")
    )
    expected = {q: r["status"] for q, r in ledger["units"].items()}
    assert shipped["states_by_unit"] == expected


@pytest.mark.parametrize("state", STATES)
def test_each_state_is_reachable_through_the_api(ledger, state):
    """Anti-vacuity: without this, every assertion below could pass on an API that
    only ever answers one thing.

    A STATE WITH NO MEMBERS IS STILL A STATE, and this test raised StopIteration on
    2026-09-27, the moment the BGL3 campaign closed the last open defect and emptied
    FIX PENDING. The anti-vacuity purpose is kept rather than skipped: for a populated
    state the API must answer it from a live unit, and for a declared-but-empty state
    the API must still KNOW the state, answering a share of 0.0 instead of denying that
    it exists. The second half is the stronger check, and it caught a real defect:
    SurfaceStatus.share read its vocabulary off `counts`, so an emptied state began
    raising "is not a published status state".
    """
    example = next((q for q, r in ledger["units"].items() if r["status"] == state), None)
    if example is not None:
        assert vfairness.status(example).state == state
        assert vfairness.status().share(state) > 0.0
        return
    assert ledger["counts"].get(state, 0) == 0, f"no unit is in {state} yet the ledger counts some"
    assert state in ledger["states"], (
        f"{state} has no members AND is not declared, so it is not a state at all"
    )
    # Declared and empty: a real 0.0, never a refusal.
    assert vfairness.status().share(state) == 0.0


def test_share_refuses_a_state_that_does_not_exist():
    """The NEGATIVE case for share(), which nothing asserted until 2026-09-27.

    share() was widened that day so a DECLARED state with no members answers 0.0
    instead of raising, because emptying FIX PENDING had made the API deny the state
    existed. Sabotaging that widening to accept ANY name left every test green: the
    property that an unrecognised state must refuse, which was itself a deliberate fix
    ("there is no honest numeric answer to what share is in a state that does not
    exist"), had no test of its own. A widening with no negative case is how a refusal
    quietly becomes a 0.0 for everything.
    """
    surface = vfairness.status()
    for bogus in ("MADE UP", "checkd", "pending", ""):
        with pytest.raises(ValueError) as excinfo:
            surface.share(bogus)
        message = str(excinfo.value)
        assert "not a published status state" in message, message
        # The message must list the DECLARED vocabulary, not whichever states happen
        # to have members today. It used to say "The three states are" beside a list
        # of two.
        for state in STATES:
            assert state in message, f"{state} missing from {message}"


def test_share_accepts_every_declared_state_however_it_is_spelled():
    """Case and separator insensitivity, across all three states including an empty
    one. The original defect here was share("checked") returning a confident 0.0 for
    874 units it had just printed, because the keys are upper case."""
    surface = vfairness.status()
    for state in STATES:
        for spelling in (
            state,
            state.lower(),
            state.replace(" ", "_").lower(),
            state.replace(" ", "-"),
        ):
            value = surface.share(spelling)
            assert 0.0 <= value <= 1.0, (spelling, value)


def test_a_class_answers_for_the_methods_reached_through_it(ledger):
    units = ledger["units"]
    classes = [
        q
        for q, r in units.items()
        if r["kind"] == "class" and any(k.startswith(q + ".") for k in units)
    ]
    assert classes, "no class on the surface has methods, so this rule is untestable"
    for qual in classes[:40]:
        name = qual.split(".")[-1]
        result = vfairness.status(name)
        methods = [k for k in units if k.startswith(qual + ".")]
        assert len(result.units) >= 1 + len(methods) or len(result.units) > 1, (
            f"{name} reported {len(result.units)} units, ignoring {len(methods)} methods"
        )
        worst = next(
            s
            for s in ("FIX PENDING", "NOT CHECKED", "CHECKED")
            if s in [units[u]["status"] for u in result.units if u in units]
        )
        assert result.state == worst, (
            f"{name} reported {result.state} while the worst of its units is {worst}"
        )


def test_a_name_nobody_can_resolve_raises_rather_than_reading_as_unchecked():
    with pytest.raises(UnknownCapabilityError) as excinfo:
        vfairness.status("this_name_is_not_a_capability_anywhere")
    assert "not on the public surface" in str(excinfo.value)


def test_a_callable_can_be_asked_about_directly(ledger):
    fn = vfairness.demographic_parity_difference
    assert vfairness.status(fn).state in STATES


# EVERY STATE, WHATEVER THE LIBRARY'S OWN STATE IS. On 2026-10-01 the last NOT
# CHECKED unit was examined, and the two tests below, which borrowed a NOT CHECKED
# unit from the live ledger to prove the API refuses one, had nothing left to
# borrow. Emptying a state must not empty the proof that the API tells states
# apart, so they now inject their own: a deep copy of the shipped status data with
# one real unit set to NOT CHECKED and another to FIX PENDING. The live ledger is
# still the subject of every other test in this file.
# The injected states come from tests/conftest.py (status_with_every_state).


def test_require_checked_passes_for_a_checked_capability(ledger):
    checked = next(q for q, r in ledger["units"].items() if r["status"] == "CHECKED")
    assert vfairness.require_checked(checked.split(".")[-1]) is None


def test_require_checked_raises_and_names_every_failure(ledger, status_with_every_state):
    not_checked = [
        q.split(".")[-1]
        for q, st in status_with_every_state["states_by_unit"].items()
        if st == "NOT CHECKED"
    ][:2]
    checked = next(
        q for q, st in status_with_every_state["states_by_unit"].items() if st == "CHECKED"
    ).split(".")[-1]
    with pytest.raises(RuntimeError) as excinfo:
        vfairness.require_checked(checked, *not_checked)
    message = str(excinfo.value)
    for name in not_checked:
        assert name in message, "require_checked stopped at the first failure"
    assert checked not in message.split(":", 1)[1].split("\n", 1)[0]


def test_checked_is_true_only_for_checked(ledger, status_with_every_state):
    """`checked` must be True for CHECKED and False for the other two.

    Skips a state with no members rather than raising StopIteration on it, and asserts
    that at least two states WERE exercised, so the rule cannot be satisfied by an API
    that only ever sees one state. FIX PENDING emptied on 2026-09-27.
    """
    seen = []
    for state in STATES:
        example = next(
            (q for q, st in status_with_every_state["states_by_unit"].items() if st == state), None
        )
        if example is None:
            continue
        seen.append(state)
        assert vfairness.status(example).checked is (state == "CHECKED")
    assert "CHECKED" in seen, "no CHECKED unit exists, so the True case was never tried"
    assert len(seen) >= 2, (
        f"only {seen} had members, so `checked` was never shown to DISTINGUISH states"
    )


def test_the_api_and_the_published_badge_agree(ledger):
    """Both are renderings of one measurement, so they are compared against the SAME
    rule rather than against each other's output: worst-of over the units a name
    covers, in the order FIX PENDING > NOT CHECKED > CHECKED."""
    page = ROOT / "docs" / "site" / "api-reference" / "index.html"
    import re

    text = page.read_text(encoding="utf-8")
    pattern = re.compile(
        r'api-function-name">([^<]+)</span><!--cap-status--><a [^>]*data-cap-status="([A-Z ]+)"'
    )
    pairs = pattern.findall(text)
    assert len(pairs) > 50, f"only {len(pairs)} badges found; the page shape changed"
    disagreements = []
    for shown, badge_state in pairs:
        name = re.sub(r"^(class|def)\s+", "", shown.strip()).lstrip("@").split("(")[0].strip()
        name = name.split(".")[-1]
        try:
            api_state = vfairness.status(name).state
        except UnknownCapabilityError:
            continue
        if api_state != badge_state:
            disagreements.append(f"{name}: page says {badge_state}, status() says {api_state}")
    assert not disagreements, "\n".join(disagreements)
