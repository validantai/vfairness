"""BGL4 audit of batch `A-_toplevel-1`: the two grades that did not hold.

WHAT THIS FILE IS. An AUDITOR's demonstration, CONVERTED TO A PIN on 2026-09-27
(BGL5) when both defects were closed. Each test asserted the behaviour the graded
unit CLAIMS and carried ``xfail(strict=True)`` because the unit did not have it;
the fixes made them XPASS, which ``xfail_strict`` turns red, so the markers are
gone and the assertions now stand as the record that the corrected behaviour
holds. The subject of each test and the measurement that found it are unchanged.

Both were measured on 2026-09-27 on this repo, BEFORE the fix. The fixes are in
``status._resolve`` (the class fold moved ABOVE ``if name in units: return
[name]``) and ``SurfaceStatus.share`` (an empty surface and an unaccounted
remainder now raise instead of answering 0.0), with the full before/after in the
comments there. Further pins, controls and the sabotage record are in
tests/test_bgl5_toplevel_and_net.py.

DEFECT 1, ``vfairness.status.status`` (graded PROVEN, bgl3). ``_resolve`` folds a
class's methods into the class so that the API and the published badge answer the
same question the same way, and its comment names ``FairnessAnalyzer`` and says
"reporting only the class row here would have said CHECKED". The fold SAT BELOW
``if name in units: return [name]``, so the FULLY QUALIFIED name and a CLASS
OBJECT (the spelling ``status()``'s own docstring invites) never reached it:

    status('vfairness.evaluation.vfairness_metrics.analyzer.FairnessAnalyzer')
        -> CHECKED, checked=True, units=1
    status('FairnessAnalyzer')       -> NOT CHECKED, units=15
    status(vfairness.FairnessAnalyzer) -> CHECKED
    require_checked(vfairness.FairnessAnalyzer) -> None, i.e. the gate PASSED

``FairnessAnalyzer.report_to_svg`` is NOT CHECKED in the shipped ledger, and
``scripts/stamp_status_badges.index_by_name`` folds unconditionally, so the badge
said NOT CHECKED for the name the API called CHECKED. 94 ledger rows were in this
shape when the audit measured it, 98 on the day it was fixed.

DEFECT 2, ``vfairness.status.SurfaceStatus.share`` (graded PROVEN, wave4). The
method refuses a state that does not exist, which is what it was graded on, and
that guard is load-bearing. It did NOT refuse a surface with nothing in it:
``(count / self.total) if self.total else 0.0`` answered a confident 0.0 for every
state of an empty surface, where the honest answer is "no surface to take a share
of". Same shape for a ``counts`` that did not account for ``total``. Neither is
reachable from the shipped ledger, so it was a unit-boundary hole in a PUBLIC,
top-level-exported class rather than a live wrong number.
"""

from __future__ import annotations

import importlib

import pytest

import vfairness

status_module = importlib.import_module("vfairness.status")

_ANALYZER = "vfairness.evaluation.vfairness_metrics.analyzer.FairnessAnalyzer"


# ─────────────────────────────────────────────────────────────────────────────
# THE CONDITION IS INJECTED, NOT FOUND IN TODAY'S LEDGER
# ─────────────────────────────────────────────────────────────────────────────
#
# These tests ask whether a CLASS answers for its methods: the fold must report the
# worst child, and the gate must refuse a class with an unchecked method behind it.
# Until 2026-09-30 they read that condition straight out of the live ledger, where
# FairnessAnalyzer.report_to_svg happened to be NOT CHECKED. The first-examination
# wave then graded every one of FairnessAnalyzer's 14 methods, the condition stopped
# existing, and all of them failed ON GOOD NEWS. Measured that day: only two classes
# in the whole library still held a worse method behind a CHECKED row, and both were
# helpers about to be graded, after which there would be NONE.
#
# A test whose fixture is "the library still has this defect" will fail the day the
# library is finished, and the cheapest way back to green is to un-grade something.
# That is a guard arguing for breaking the product. So the condition is now
# constructed: a copy of the real ledger with ONE method forced to NOT CHECKED. The
# real resolver, the real surface and the real class are all still exercised; only
# the state of that one method is guaranteed instead of hoped for.
@pytest.fixture
def ledger_with_an_unchecked_method(monkeypatch):
    import copy

    status_module._cache = None
    real = status_module._load()
    fixed = copy.deepcopy(real)
    target = _ANALYZER + ".report_to_svg"
    assert target in fixed["states_by_unit"], "the injected method no longer exists"
    fixed["states_by_unit"][target] = "NOT CHECKED"
    monkeypatch.setattr(status_module, "_cache", fixed)
    yield fixed
    status_module._cache = None


def _ledger() -> dict:
    return status_module._load()["states_by_unit"]


def test_the_class_whose_method_is_unchecked_is_not_reported_checked(
    ledger_with_an_unchecked_method,
):
    """CONTROL, and it passes: the ledger really does hold a worse method behind a
    CHECKED class row, so the two tests below are asking about real data."""
    ledger = _ledger()
    assert ledger[_ANALYZER] == "CHECKED"
    children = {q: s for q, s in ledger.items() if q.startswith(_ANALYZER + ".")}
    assert children, "no methods behind the class: the fixture would be vacuous"
    assert "NOT CHECKED" in set(children.values()), children


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: BGL4 audit: the class fold in
# _resolve sits below `if name in units: return [name]`, so a fully qualified class name and a
# class object skip it
def test_a_class_answers_for_its_methods_however_the_name_is_spelled(
    ledger_with_an_unchecked_method,
):
    """The three spellings of one question must not disagree.

    The tail spelling folds and answers NOT CHECKED. The fully qualified name and
    the class object report CHECKED for the same class, so a caller who follows
    the docstring and passes the object gets the opposite answer.
    """
    # THE EXPECTED STATE IS DERIVED, NEVER QUOTED. It was the literal "NOT CHECKED"
    # until 2026-09-30, and the tier-1 audit wave then overturned two of this class's
    # own methods to DEFECT OPEN, so the fold correctly began answering FIX PENDING
    # and all three spellings agreed on it. The test failed anyway, because the
    # literal was the state that happened to be true the day it was written. The
    # SUBJECT is that the three spellings must not disagree, and that the fold takes
    # the WORST child; neither depends on which state that is. Deriving it from the
    # ledger through the module's own ordering keeps this test alive across every
    # future wave, instead of turning a correct re-grade into a red test whose
    # cheapest fix is to undo the re-grade.
    children = [s for q, s in _ledger().items() if q.startswith(_ANALYZER + ".")]
    assert children, "no methods behind the class: the fold would be vacuous"
    expected = next(s for s in status_module._WORST_FIRST if s in set(children))

    by_tail = vfairness.status("FairnessAnalyzer")
    by_qualified = vfairness.status(_ANALYZER)
    by_object = vfairness.status(vfairness.FairnessAnalyzer)

    assert by_tail.state == expected, (
        f"the fold answered {by_tail.state} where the worst of its "
        f"{len(children)} method state(s) is {expected}"
    )
    assert by_qualified.state == by_tail.state, (
        f"status({_ANALYZER!r}) says {by_qualified.state} over "
        f"{len(by_qualified.units)} unit(s) while status('FairnessAnalyzer') says "
        f"{by_tail.state} over {len(by_tail.units)}"
    )
    assert by_object.state == by_tail.state, (
        f"status(FairnessAnalyzer) says {by_object.state}, the tail says {by_tail.state}"
    )


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: BGL4 audit: require_checked
# inherits the unfolded class row, so the CI gate passes on a class with a NOT CHECKED method
def test_the_gate_does_not_pass_a_class_with_an_unchecked_method(ledger_with_an_unchecked_method):
    """``require_checked`` promises to raise unless every name is CHECKED. Asked by
    the class object, it returns None while ``report_to_svg`` behind that class is
    NOT CHECKED, which is the same collapse of "nothing was examined" into
    "nothing was found" that its own empty-call guard exists to prevent."""
    with pytest.raises(RuntimeError):
        vfairness.require_checked(vfairness.FairnessAnalyzer)


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: BGL4 audit: share() answers
# 0.0 for a surface of zero units instead of refusing; 0/0 is a could-not-answer, not a share
def test_the_share_of_an_empty_surface_is_refused_not_answered_with_zero():
    """A surface with no code units in it has no share to report. The method's own
    docstring says there is no honest numeric answer to a question about a state
    that does not exist; there is equally none about a population that does not."""
    empty = status_module.SurfaceStatus(
        total=0, counts={}, evidence_dates=(), library_version="0.1.0"
    )
    with pytest.raises(ValueError):
        empty.share("CHECKED")


# WAS xfail(strict=True) until the BGL5 fix. The recorded reason: BGL4 audit: share() reads
# absent-from-counts as zero even when counts does not account for total, so the unmeasured
# remainder reports as 0.0
def test_a_counts_that_does_not_account_for_total_is_refused():
    """1,307 of 1,567 units accounted for leaves 260 in no state at all. Answering
    0.0 for NOT CHECKED asserts a tally of the 260 that nothing supplied."""
    partial = status_module.SurfaceStatus(
        total=1567, counts={"CHECKED": 1307}, evidence_dates=(), library_version="0.1.0"
    )
    with pytest.raises(ValueError):
        partial.share("NOT CHECKED")
