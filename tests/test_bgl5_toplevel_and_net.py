"""BGL5 pins: the seven defects an independent audit overturned in four batches.

WHAT THIS FILE IS. The fix side of the BGL4 audit of batches ``A-_toplevel-1``,
``A-_toplevel-2``, ``A-net-1`` and ``A-xai-2``. Every test here asserts a
behaviour that was MEASURED to be wrong on 2026-09-27 and is now right, and every
one of them was reddened by restoring its defect in memory before being believed.
Each section names the unit, what it returned before, and what it returns now.

The controls matter as much as the pins: a fix that refuses everything passes every
refusal test and destroys the library, so each section also asserts that healthy
input still gets its real number, with the number written down.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import warnings

import numpy as np
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


# ---------------------------------------------------------------------------
# A-_toplevel-1, defect 1: status() and require_checked() mis-resolved a CLASS.
#
# The class fold in status._resolve sat BELOW `if name in units: return [name]`,
# so it was dead for a fully qualified name and for a class object. Measured
# before the fix:
#     status(_ANALYZER).state                     -> 'CHECKED'  (1 unit)
#     status('FairnessAnalyzer').state            -> 'NOT CHECKED' (15 units)
#     status(vfairness.FairnessAnalyzer).state    -> 'CHECKED'
#     require_checked(vfairness.FairnessAnalyzer) -> None, the gate PASSED
# After: all three spellings resolve 15 units and answer NOT CHECKED, and the
# gate raises RuntimeError. One fix closes both graded rows.
# ---------------------------------------------------------------------------


def test_the_ledger_still_holds_a_worse_method_behind_a_checked_class(
    ledger_with_an_unchecked_method,
):
    """ANTI-VACUITY, and it must pass on its own: without a class row that is
    CHECKED while a method behind it is not, the two pins below would be asking
    about nothing."""
    ledger = _ledger()
    assert ledger[_ANALYZER] == "CHECKED"
    children = {q: s for q, s in ledger.items() if q.startswith(_ANALYZER + ".")}
    assert children, "no methods behind the class: the fixture would be vacuous"
    assert "NOT CHECKED" in set(children.values()), children


def test_a_class_answers_for_its_methods_however_the_name_is_spelled(
    ledger_with_an_unchecked_method,
):
    """Three spellings of one question, one answer.

    The tail spelling folded and answered NOT CHECKED; the fully qualified name and
    the class object skipped the fold and answered CHECKED, so a caller who followed
    status()'s own docstring and passed the object got the opposite answer from the
    one the published badge gives.
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
    assert len(by_qualified.units) == len(by_tail.units) == len(by_object.units)
    assert len(by_tail.units) > 1, "the fold reached no methods at all"
    assert by_object.checked is False


def test_the_gate_does_not_pass_a_class_with_an_unchecked_method(ledger_with_an_unchecked_method):
    """``require_checked`` promises to raise unless every name is CHECKED. Asked by
    the class object it returned None while ``report_to_svg`` behind that class is
    NOT CHECKED, the same collapse of "nothing was examined" into "nothing was
    found" that its own empty-call guard exists to prevent."""
    for spelling in (vfairness.FairnessAnalyzer, _ANALYZER, "FairnessAnalyzer"):
        with pytest.raises(RuntimeError) as excinfo:
            vfairness.require_checked(spelling)
        assert "not CHECKED" in str(excinfo.value)


def test_the_fold_does_not_refuse_a_class_whose_methods_are_all_checked():
    """OVER-CORRECTION CONTROL. A fold that made every class answer NOT CHECKED
    would pass the pin above and make the gate useless. 141 class rows in the
    shipped ledger are CHECKED with every method CHECKED; the gate must still pass
    for them, by all three spellings."""
    ledger = _ledger()
    fully_checked = None
    for qual, state in ledger.items():
        kids = [k for k in ledger if k.startswith(qual + ".")]
        tail = qual.split(".")[-1]
        namesakes = [k for k in ledger if k.split(".")[-1] == tail]
        if kids and len(namesakes) == 1 and state == "CHECKED":
            if all(ledger[k] == "CHECKED" for k in kids):
                fully_checked = (qual, tail, len(kids))
                break
    assert fully_checked, "no all-CHECKED class with methods: the control is vacuous"
    qual, tail, n_kids = fully_checked
    for spelling in (qual, tail):
        result = vfairness.status(spelling)
        assert result.state == "CHECKED", (spelling, result.state, result.detail)
        assert result.checked is True
        assert len(result.units) == n_kids + 1, (spelling, result.units)
        assert vfairness.require_checked(spelling) is None


def test_a_fully_qualified_leaf_still_means_itself_alone(ledger_with_an_unchecked_method):
    """OVER-CORRECTION CONTROL. The fold must not turn a precise question into a
    vague one: a fully qualified function name resolves to exactly one unit and
    keeps its own detail string, and a method spelled in full does not answer for
    its siblings."""
    leaf = vfairness.status("vfairness._names.name_token_list")
    assert len(leaf.units) == 1
    assert leaf.state == _ledger()["vfairness._names.name_token_list"]
    assert "code units behind this name" not in leaf.detail

    method = vfairness.status(_ANALYZER + ".report_to_svg")
    assert len(method.units) == 1
    assert method.state == "NOT CHECKED"


def test_the_installed_api_and_the_badge_rule_agree_on_every_class():
    """The rule this fix exists for: the API and ``stamp_status_badges.index_by_name``
    must fold the same way. Checked against the badge generator's rule computed here
    from the ledger, over every class row, for BOTH spellings."""
    ledger = _ledger()
    worst_first = ("FIX PENDING", "NOT CHECKED", "CHECKED")
    checked_classes = 0
    for qual, state in ledger.items():
        kids = [k for k in ledger if k.startswith(qual + ".")]
        if not kids:
            continue
        checked_classes += 1
        states = [state] + [ledger[k] for k in kids]
        expected = next(s for s in worst_first if s in states)
        assert vfairness.status(qual).state == expected, qual
    assert checked_classes > 100, f"only {checked_classes} class rows folded"


# ---------------------------------------------------------------------------
# A-_toplevel-1, defect 2: SurfaceStatus.share fabricated a zero.
#
# Measured before the fix:
#     SurfaceStatus(total=0, counts={}).share('CHECKED')             -> 0.0
#     SurfaceStatus(total=0, counts={'CHECKED': 5}).share('CHECKED') -> 0.0
#     SurfaceStatus(total=1567, counts={'CHECKED': 1307})
#         .share('NOT CHECKED')                                     -> 0.0
#         and its three shares summed to 0.834, not 1.0
# After: each raises ValueError. A declared state that is legitimately empty still
# returns 0.0 and an undeclared state still raises, which is the behaviour the
# pins of 2026-09-25 and 2026-09-27 established and this fix had to preserve.
# ---------------------------------------------------------------------------


def test_the_share_of_an_empty_surface_is_refused_not_answered_with_zero():
    """A surface with no code units in it has no share to report. share()'s own
    docstring says there is no honest numeric answer about a state that does not
    exist; there is equally none about a population that does not."""
    empty = status_module.SurfaceStatus(
        total=0, counts={}, evidence_dates=(), library_version="0.1.0"
    )
    for state in ("CHECKED", "NOT CHECKED", "FIX PENDING", "checked"):
        with pytest.raises(ValueError) as excinfo:
            empty.share(state)
        assert "no population" in str(excinfo.value), str(excinfo.value)

    contradictory = status_module.SurfaceStatus(
        total=0, counts={"CHECKED": 5}, evidence_dates=(), library_version="0.1.0"
    )
    with pytest.raises(ValueError):
        contradictory.share("CHECKED")


def test_a_counts_that_does_not_account_for_total_is_refused():
    """1,307 of 1,567 units accounted for leaves 260 in no state at all. Answering
    0.0 for NOT CHECKED asserts a tally of those 260 that nothing supplied."""
    partial = status_module.SurfaceStatus(
        total=1567, counts={"CHECKED": 1307}, evidence_dates=(), library_version="0.1.0"
    )
    with pytest.raises(ValueError) as excinfo:
        partial.share("NOT CHECKED")
    message = str(excinfo.value)
    assert "1307 of 1567" in message, message
    assert "260" in message, message
    # The state that IS tallied keeps its real fraction: 1307/1567.
    assert partial.share("CHECKED") == pytest.approx(1307 / 1567)


def test_the_live_surface_still_reports_its_real_shares():
    """OVER-CORRECTION CONTROL, with the actual numbers. The shipped ledger is
    consistent, so every declared state still answers: CHECKED 1268/1567 =
    0.80919, NOT CHECKED 299/1567 = 0.19081, FIX PENDING an empty but real 0.0,
    and the three sum to exactly 1.0."""
    surface = vfairness.status()
    assert surface.total == sum(surface.counts.values())
    for state, count in surface.counts.items():
        assert surface.share(state) == pytest.approx(count / surface.total)
        assert surface.share(state) > 0.0
    for state in ("FIX PENDING", "NOT CHECKED", "CHECKED"):
        if state not in surface.counts:
            # Declared and empty is a measurement, and the happiest one here.
            assert surface.share(state) == 0.0
    assert sum(surface.share(s) for s in ("FIX PENDING", "NOT CHECKED", "CHECKED")) == (
        pytest.approx(1.0)
    )
    assert surface.share("checked") == surface.share("CHECKED")
    assert surface.share("not_checked") == surface.share("NOT CHECKED")
    with pytest.raises(ValueError, match="not a published status state"):
        surface.share("verified")


# ---------------------------------------------------------------------------
# A-_toplevel-2, defect 3: branding.set_branding resolved what it could not read.
#
# Measured before the fix, each token through BOTH documented routes:
#     set_branding('off')      -> branding_enabled() True,  no warning
#     set_branding('0')        -> True,  set_branding('false') -> True
#     set_branding('no')       -> True,  set_branding('none')  -> True
#     set_branding('OFF')      -> True
#     VFAIRNESS_BRANDING='off' -> branding_enabled() False
#     set_branding('')         -> False, no warning (the env route warns and
#                                 leaves branding ON for the same value)
# After: one vocabulary for both routes, and a value neither can read leaves the
# switch untouched and warns.
# ---------------------------------------------------------------------------


@pytest.fixture()
def branding_module(monkeypatch):
    module = importlib.import_module("vfairness.branding")
    monkeypatch.delenv(module.ENV_VAR, raising=False)
    module.set_branding(None)
    try:
        yield module
    finally:
        module.set_branding(None)


def _warned(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught]


@pytest.mark.parametrize("token", ["off", "0", "false", "no", "none", "OFF", " none "])
def test_set_branding_reads_the_documented_disabling_vocabulary(branding_module, token):
    """The two steps of one resolution order must not answer a word differently.

    Every token here is listed by the module docstring as DISABLING branding, and
    every one of them silently ENABLED it through the setter.
    """
    branding_module.set_branding(None)
    enabled, msgs = _warned(
        lambda: (branding_module.set_branding(token), branding_module.branding_enabled())[1]
    )
    assert enabled is False, (token, enabled)
    assert msgs == [], (token, msgs)

    # The identical token through the environment, for comparison, not by proxy.
    branding_module.set_branding(None)
    import os

    os.environ[branding_module.ENV_VAR] = token
    try:
        via_env, env_msgs = _warned(branding_module.branding_enabled)
    finally:
        del os.environ[branding_module.ENV_VAR]
    assert via_env is enabled, (token, via_env, enabled)
    assert env_msgs == [], (token, env_msgs)


@pytest.mark.parametrize("token", ["on", "1", "true", "YES", " On "])
def test_set_branding_reads_the_documented_enabling_vocabulary(branding_module, token):
    """The other half of the one vocabulary, so the fix is not a one-way street:
    the truthy words keep the mark on, silently, and the environment route agrees."""
    branding_module.set_branding(None)
    enabled, msgs = _warned(
        lambda: (branding_module.set_branding(token), branding_module.branding_enabled())[1]
    )
    assert enabled is True, (token, enabled)
    assert msgs == [], (token, msgs)

    branding_module.set_branding(None)
    import os

    os.environ[branding_module.ENV_VAR] = token
    try:
        via_env, env_msgs = _warned(branding_module.branding_enabled)
    finally:
        del os.environ[branding_module.ENV_VAR]
    assert via_env is enabled, (token, via_env, enabled)
    assert env_msgs == [], (token, env_msgs)


@pytest.mark.parametrize("token", ["", "   ", "disable", "no thanks", "2", 2, -1, [], [0], {}])
def test_set_branding_discloses_a_value_it_cannot_read(branding_module, token):
    """A setter may not resolve what it cannot read, in either direction.

    '' used to strip the mark in silence, which is the direction that matters: the
    module exists so a client deliverable can ship without our name on it, and a
    typo deciding that question either way with no word said is the defect.
    """
    branding_module.set_branding(None)
    enabled, msgs = _warned(
        lambda: (branding_module.set_branding(token), branding_module.branding_enabled())[1]
    )
    assert enabled is True, (token, enabled)
    assert any("is not a value this switch can read" in m for m in msgs), (token, msgs)
    assert any("NOT changed" in m for m in msgs), (token, msgs)


def test_an_unreadable_value_does_not_undo_a_deliberate_switch(branding_module):
    """The refusal leaves the switch EXACTLY as it was, so a typo cannot silently
    re-brand output a caller deliberately unbranded."""
    branding_module.set_branding(False)
    assert branding_module.branding_enabled() is False
    _, msgs = _warned(lambda: branding_module.set_branding("disable"))
    assert any("still OFF" in m for m in msgs), msgs
    assert branding_module.branding_enabled() is False


def test_the_branding_switch_still_switches(branding_module):
    """OVER-CORRECTION CONTROL. A setter that refused everything would pass every
    test above and make the documented switch unusable."""
    # Only the values the PRE-FIX setter also got right, so this control stays green
    # under the sabotage and is therefore a control and not a second pin. The string
    # vocabulary is pinned above, where it belongs.
    for value, expected in (
        (True, True),
        (False, False),
        (0, False),
        (1, True),
        (np.True_, True),
        (np.False_, False),
    ):
        branding_module.set_branding(None)
        enabled, msgs = _warned(
            lambda v=value: (
                branding_module.set_branding(v),
                branding_module.branding_enabled(),
            )[1]
        )
        assert enabled is expected, (value, enabled)
        assert msgs == [], (value, msgs)
    # None still clears the override back to the environment.
    import os

    os.environ[branding_module.ENV_VAR] = "off"
    try:
        branding_module.set_branding(True)
        assert branding_module.branding_enabled() is True
        branding_module.set_branding(None)
        assert branding_module.branding_enabled() is False
    finally:
        del os.environ[branding_module.ENV_VAR]


# ---------------------------------------------------------------------------
# A-_toplevel-2, defect 4: streaming_demographic_parity hid a vanished group.
#
# A group leaves the comparison by TWO routes and the earlier fix covered only
# min_group_size. Measured before this fix on 140 rows, 'a' and 'b' 50 rows each
# selected at 0.5 and 'c' 40 rows whose predictions are ALL NaN:
#     streaming_demographic_parity(...)      -> 0.0, ONE warning, a row count,
#                                               'c' named nowhere
#     demographic_parity_difference(...)     -> 0.0 AND "Group(s) c lost every row
#                                               and are absent from every
#                                               comparison computed from this data"
# After: the streaming twin emits that same sentence, at both surfaces, at every
# chunk size, and the fail-closed warning counts the groups the DATA held (it
# said "out of 2" for three groups).
# ---------------------------------------------------------------------------


def _vanished_group_fixture():
    """140 rows, three groups. 'a' and 'b' are 50 rows each selected at 0.5. 'c'
    is 40 rows whose predictions are ALL NaN, so its selection rate cannot be
    measured at all: a classifier that abstained on one subpopulation."""
    y_pred = np.array([1.0] * 25 + [0.0] * 25 + [1.0] * 25 + [0.0] * 25 + [np.nan] * 40)
    sensitive = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 40, dtype=object)
    return y_pred, sensitive


_LOST = "lost every row and are absent from every comparison computed from this data"


def test_streaming_dp_discloses_a_group_that_lost_every_row():
    """What the PROVEN grade claimed was already true.

    The value stays 0.0 because it must keep matching the whole-array result; what
    was missing was saying that a whole group is absent from it.
    """
    from vfairness.streaming import streaming_demographic_parity

    y_pred, sensitive = _vanished_group_fixture()
    value, msgs = _warned(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert value == 0.0
    disclosure = [m for m in msgs if _LOST in m]
    assert disclosure, msgs
    assert "'c'" in disclosure[0], disclosure
    assert "could not check" in disclosure[0], disclosure


def test_the_streaming_twin_and_the_core_name_the_same_vanished_group():
    """Two renderings of one measurement must not disagree about how much of the
    data they looked at. The core named the group; the streaming twin did not."""
    from vfairness.evaluation.vfairness_metrics.classification import (
        demographic_parity_difference,
    )
    from vfairness.streaming import streaming_demographic_parity

    y_pred, sensitive = _vanished_group_fixture()
    core_value, core_msgs = _warned(
        lambda: demographic_parity_difference(np.zeros(140), y_pred, sensitive)
    )
    stream_value, stream_msgs = _warned(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert core_value == pytest.approx(stream_value) == 0.0
    for msgs in (core_msgs, stream_msgs):
        assert any(_LOST in m for m in msgs), msgs
        assert any("c" in m and _LOST in m for m in msgs), msgs


@pytest.mark.parametrize("chunk_size", [1, 3, 40, 139, 140, 100_000])
def test_the_vanished_group_is_named_at_every_chunk_size(chunk_size):
    """The disclosure is an aggregation like the counts, so it must be
    chunk-invariant: the audit measured group_c_named=False at all six sizes."""
    from vfairness.streaming import streaming_demographic_parity

    y_pred, sensitive = _vanished_group_fixture()
    value, msgs = _warned(
        lambda: streaming_demographic_parity(y_pred, sensitive, chunk_size=chunk_size)
    )
    assert value == 0.0
    assert any("'c'" in m and _LOST in m for m in msgs), (chunk_size, msgs)


def test_stream_group_counts_names_a_group_that_lost_every_row():
    """The counts dict cannot carry it either: 'c' is not a key, so the warning is
    the only channel. It said "40 of 140 rows were excluded" and no group name."""
    from vfairness.streaming import stream_group_counts

    y_pred, sensitive = _vanished_group_fixture()
    counts, msgs = _warned(lambda: stream_group_counts(y_pred, sensitive))
    assert set(counts) == {"a", "b"}
    assert any("'c'" in m and _LOST in m for m in msgs), msgs


def test_the_fail_closed_warning_counts_the_groups_in_the_data():
    """It reported "out of {len(counts)}", which undercounts for the same reason:
    a group with no qualifying row is not in counts. Three groups in the data, one
    of them all-NaN, one below the gate."""
    from vfairness.streaming import streaming_demographic_parity

    y_pred = np.array([np.nan] * 50 + [1.0] * 25 + [0.0] * 25 + [0.0] * 10)
    sensitive = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 10, dtype=object)
    value, msgs = _warned(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert np.isnan(value)
    closed = [m for m in msgs if "no between-group comparison" in m]
    assert closed, msgs
    assert "out of 3 group(s) in the data" in closed[0], closed
    assert "'a'" in closed[0], closed


def test_the_streaming_metric_still_measures_a_clean_run_in_silence():
    """OVER-CORRECTION CONTROL, with the actual numbers. Two full groups selected
    at 0.8 and 0.2 give a measured gap of 0.6 with NO warning, and the size-gate
    disclosure still fires on its own fixture and still lets the gap through when
    the gate is lowered (0.0 at min_group_size=30, 0.5 at 10)."""
    from vfairness.streaming import stream_group_counts, streaming_demographic_parity

    y_pred = np.array([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40)
    sensitive = np.array(["a"] * 50 + ["b"] * 50, dtype=object)
    value, msgs = _warned(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert value == pytest.approx(0.6)
    assert msgs == [], msgs

    counts, count_msgs = _warned(lambda: stream_group_counts(y_pred, sensitive))
    assert counts == {"a": (40, 50), "b": (10, 50)}
    assert count_msgs == [], count_msgs

    gated_pred = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25 + [0] * 10)
    gated_sens = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 10, dtype=object)
    gated, gated_msgs = _warned(lambda: streaming_demographic_parity(gated_pred, gated_sens))
    assert gated == 0.0
    assert any("excluding 1 group" in m and "'c': 10" in m for m in gated_msgs), gated_msgs
    assert not any(_LOST in m for m in gated_msgs), "no group lost every row here"
    lowered, _ = _warned(
        lambda: streaming_demographic_parity(gated_pred, gated_sens, min_group_size=10)
    )
    assert lowered == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# A-net-1: GuardedSession.close. The overturn itself is a REGRADE (SEMI-PROVEN ->
# NOT A MEASUREMENT: the unit takes no measurable input, returns None on every
# session state and gates no measurement), which lives in the grading records and
# not in this file. What WAS a code defect in it is the robustness gap the audit
# found while sweeping the state classes.
#
# Measured before the fix, on a session holding one raising adapter and one healthy
# one, with a recording adapter mounted on the parent session:
#     RuntimeError('poolmanager gone') propagated
#     len(session._guard_adapters) == 2          (the loop aborted, nothing cleared)
#     the parent session's adapter was NOT closed (super().close() unreached)
# After: the same RuntimeError propagates, the cache is empty, and the parent
# session's adapter IS closed.
# ---------------------------------------------------------------------------


def _pinned_adapter():
    from vfairness.net.egress import PinnedIPAdapter

    return PinnedIPAdapter("localhost", "127.0.0.1", tls=False)


def test_close_closes_every_adapter_even_when_one_of_them_raises():
    """A failing close() may not decide that the remaining sockets stay open.

    The exception still reaches the caller, because a failing close is a real
    failure and swallowing it would be the opposite defect.
    """
    from vfairness.net.egress import GuardedSession, PinnedIPAdapter

    class _Boom(PinnedIPAdapter):
        def close(self) -> None:
            raise RuntimeError("poolmanager gone")

    class _Recorder(PinnedIPAdapter):
        closed = False

        def close(self) -> None:
            type(self).closed = True
            super().close()

    _Recorder.closed = False
    session = GuardedSession(allow_http=True, allow_loopback=True)
    session._guard_adapters[("http", "boom", 80)] = _Boom("localhost", "127.0.0.1", tls=False)
    session._guard_adapters[("http", "survivor", 80)] = _pinned_adapter()
    # The observable for "super().close() was reached": requests.Session.close()
    # closes every adapter mounted on the session itself.
    session.adapters["sentinel://"] = _Recorder("localhost", "127.0.0.1", tls=False)

    with pytest.raises(RuntimeError, match="poolmanager gone"):
        session.close()
    assert session._guard_adapters == {}
    assert _Recorder.closed is True, "super().close() was never reached"


def test_a_second_failing_close_is_reported_and_not_lost():
    """Two failures must not become one silent one: the first is raised and the
    rest are named on it, after everything has been closed."""
    from vfairness.net.egress import GuardedSession, PinnedIPAdapter

    class _BoomA(PinnedIPAdapter):
        def close(self) -> None:
            raise RuntimeError("first failure")

    class _BoomB(PinnedIPAdapter):
        def close(self) -> None:
            raise ValueError("second failure")

    session = GuardedSession(allow_http=True, allow_loopback=True)
    session._guard_adapters[("http", "a", 80)] = _BoomA("localhost", "127.0.0.1", tls=False)
    session._guard_adapters[("http", "b", 80)] = _BoomB("localhost", "127.0.0.1", tls=False)
    with pytest.raises(RuntimeError) as excinfo:
        session.close()
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("second failure" in note for note in notes), notes
    assert session._guard_adapters == {}


def test_a_healthy_close_is_unchanged_and_the_guard_still_refuses():
    """OVER-CORRECTION CONTROL, and the regrade's own facts.

    A close that has nothing to complain about still returns None, warns nothing,
    empties the cache and is idempotent; and clearing the vetted-adapter cache can
    only make the next hop re-validate, so the SSRF refusal is still there after
    close, with the same message. That is why this unit gates no measurement.
    """
    from vfairness.net.egress import GuardedSession, SSRFError

    session = GuardedSession(allow_http=True, allow_loopback=True)
    session._guard_adapters[("http", "localhost", 80)] = _pinned_adapter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert session.close() is None
        assert session.close() is None
    assert [str(w.message) for w in caught] == []
    assert session._guard_adapters == {}

    guarded = GuardedSession(allow_http=True, allow_loopback=False)
    metadata = "http://169.254.169.254/latest/meta-data/"
    with pytest.raises(SSRFError) as before:
        guarded.get_adapter(metadata)
    guarded.close()
    with pytest.raises(SSRFError) as after:
        guarded.get_adapter(metadata)
    assert "non-public address" in str(before.value)
    assert str(after.value) == str(before.value)


# ---------------------------------------------------------------------------
# A-xai-2: xai.sidecar_cli.main printed a hand-listed subset of the Explanation.
#
# Measured before the fix, feeding the real CLI a CSV whose column b is all NaN:
#     the adapter computed unattributed_features=['b'], attributions_complete=False,
#     prediction_measured=False, base_value_measured=False,
#     local_accuracy_residual=None, local_accuracy_ok=None, and warned twice
#     the envelope printed {"success": true, ... "base_value": NaN, "prediction":
#     NaN, "attributions": [{"a": 2.34}, {"b": NaN}, {"c": 0.03}]} and carried NONE
#     of those six fields, so two finite contributions made it read as usable
#     node -e JSON.parse(...) -> 'FAILED: Unexpected token N ... is not valid JSON'
#     a 1-row CSV -> success=true with every contribution exactly 0.0, because
#     background = X[: min(50, len(X))] IS the explained instance
# After: the six fields and the whole params dict reach the envelope, non-finite
# numbers render as JSON null, a degenerate background is named, and the
# background-sufficiency facts are computed ABOVE the dispatch so the path an
# explicit payload['method'] takes reports them too.
# ---------------------------------------------------------------------------

_XAI_COLS = ["a", "b", "c"]
_XAI_DISCLOSURE = (
    "prediction_measured",
    "base_value_measured",
    "attributions_complete",
    "unattributed_features",
    "local_accuracy_residual",
    "local_accuracy_ok",
)


@pytest.fixture(scope="module")
def xai_model_blob():
    pytest.importorskip("shap", reason="the sidecar's explainers need shap")
    joblib = pytest.importorskip("joblib", reason="the sidecar loads models with joblib")
    import base64
    import io

    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 3))
    labels = (features[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(features, labels)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return base64.b64encode(buffer.getvalue()).decode(), model


def _xai_csv(rows) -> str:
    import math

    lines = [",".join(_XAI_COLS)]
    for row in rows:
        lines.append(
            ",".join(
                "" if (isinstance(v, float) and math.isnan(v)) else repr(float(v)) for v in row
            )
        )
    return "\n".join(lines) + "\n"


def _run_sidecar(monkeypatch, capsys, payload):
    import io

    from vfairness.xai import sidecar_cli

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    sidecar_cli.main()
    raw = capsys.readouterr().out.strip().splitlines()[-1]
    return raw, json.loads(raw)


def test_the_envelope_carries_the_disclosure_the_adapter_computed(
    monkeypatch, capsys, xai_model_blob
):
    """A correct None that the CLI drops is still a defect, one layer up.

    The producer pin (tests/test_bgl_stage2_s2g09.py, local_accuracy_residual is
    None) stayed green throughout, which is exactly why this had to be checked at
    the surface a person reads.
    """
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 3))
    features[:, 1] = np.nan
    _raw, envelope = _run_sidecar(
        monkeypatch,
        capsys,
        {
            "csv_data": _xai_csv(features),
            "feature_columns": _XAI_COLS,
            "model_base64": xai_model_blob[0],
            "trust_input": True,
            "method": "shap.LinearExplainer",
        },
    )
    explanation = envelope["data"]["explanation"]
    for key in _XAI_DISCLOSURE:
        assert key in explanation, (key, sorted(explanation))
    assert explanation["unattributed_features"] == ["b"]
    assert explanation["attributions_complete"] is False
    assert explanation["prediction_measured"] is False
    assert explanation["base_value_measured"] is False
    assert explanation["local_accuracy_residual"] is None
    assert explanation["local_accuracy_ok"] is None
    assert explanation["attributions_measured"] is False
    assert "b" in explanation["attributions_measured_reason"]
    # The adapter's own dict travels too, so the two surfaces cannot carry
    # different amounts of one disclosure.
    assert explanation["params"]["unattributed_features"] == ["b"]


def test_the_envelope_is_valid_json_for_a_parser_that_is_not_python(
    monkeypatch, capsys, xai_model_blob
):
    """Through the REAL entry point, because the contract is what the process
    prints: `echo payload | python -m vfairness.xai.sidecar_cli`.

    json.dumps wrote the bare token NaN, which RFC 8259 forbids. Python's loads
    accepts it, so a Python consumer carried the NaN onward and anything else saw a
    parse error instead of a refusal.

    The same payload is run IN PROCESS first, deliberately: a subprocess cannot be
    reached by an in-memory sabotage, so a pin that only shelled out could not be
    shown to discriminate. The in-process half proves the rendering, the subprocess
    half proves the real entry point does it too.
    """
    payload = {
        "csv_data": _xai_csv(np.full((60, 3), np.nan)),
        "feature_columns": _XAI_COLS,
        "model_base64": xai_model_blob[0],
        "trust_input": True,
        "method": "shap.LinearExplainer",
    }
    in_process_raw, in_process = _run_sidecar(monkeypatch, capsys, payload)
    for token in ("NaN", "Infinity"):
        assert token not in in_process_raw, (
            f"the envelope still contains the bare token {token}: {in_process_raw[:200]}"
        )
    assert in_process["data"]["explanation"]["prediction"] is None

    completed = subprocess.run(
        # -W always so the stderr assertion below cannot depend on the parent
        # environment's warning filters.
        [sys.executable, "-W", "always", "-m", "vfairness.xai.sidecar_cli"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    raw = completed.stdout.strip().splitlines()[-1]
    for token in ("NaN", "Infinity"):
        assert token not in raw, f"the envelope still contains the bare token {token}: {raw[:200]}"

    def _reject(constant: str) -> float:
        raise ValueError(f"not valid JSON: bare {constant}")

    envelope = json.loads(raw, parse_constant=_reject)
    explanation = envelope["data"]["explanation"]
    assert explanation["base_value"] is None
    assert explanation["prediction"] is None
    assert [a["contribution"] for a in explanation["attributions"]] == [None, None, None]
    assert explanation["prediction_measured"] is False
    # The two UserWarnings still reach stderr, where they were before.
    assert "local accuracy is UNCHECKED" in completed.stderr


@pytest.mark.parametrize("n_rows", [1, 60])
def test_a_background_that_is_the_explained_instance_is_disclosed(
    monkeypatch, capsys, xai_model_blob, n_rows
):
    """Every SHAP value is exactly 0.0 by construction here, which reads as "no
    feature mattered" for a model whose only real driver is feature a.

    n_rows=1 is the 1-row CSV the audit measured; n_rows=60 is 60 IDENTICAL rows,
    which a row count would miss, so the test is whether the background differs
    from the instance at all. Both take the explicit-method path, which skips the
    router, which is where nothing consulted background sufficiency before.
    """
    rng = np.random.default_rng(7)
    one = rng.normal(size=(1, 3))
    rows = np.repeat(one, n_rows, axis=0)
    _raw, envelope = _run_sidecar(
        monkeypatch,
        capsys,
        {
            "csv_data": _xai_csv(rows),
            "feature_columns": _XAI_COLS,
            "model_base64": xai_model_blob[0],
            "trust_input": True,
            "method": "shap.KernelExplainer",
        },
    )
    explanation = envelope["data"]["explanation"]
    assert [a["contribution"] for a in explanation["attributions"]] == [0.0, 0.0, 0.0]
    assert explanation["background_is_the_explained_instance"] is True
    assert explanation["background_rows"] == min(50, n_rows)
    assert explanation["attributions_measured"] is False
    assert "0.0 by construction" in explanation["attributions_measured_reason"]
    assert envelope["data"]["method_explicit"] is True
    # Computed above the dispatch, so it is reported on the path that never calls
    # the router: len(csv) >= 10 is False for one row and True for sixty.
    assert explanation["background_available"] is (n_rows >= 10)


def test_a_healthy_explanation_still_carries_its_real_numbers(monkeypatch, capsys, xai_model_blob):
    """OVER-CORRECTION CONTROL, with the numbers asserted against each other.

    A CLI that nulled everything, or refused whenever a disclosure field was
    absent, would pass every pin above and return nothing usable. On 200 healthy
    rows the envelope is success=true with finite numbers, the local-accuracy
    identity holds (base_value + sum(contributions) == prediction to 1e-9), the
    residual is a real measured float below the 1e-6 tolerance, and feature 'a',
    the only real driver in the fixture, carries the largest contribution.
    """
    blob, model = xai_model_blob
    rng = np.random.default_rng(11)
    rows = rng.normal(size=(200, 3))
    _raw, envelope = _run_sidecar(
        monkeypatch,
        capsys,
        {
            "csv_data": _xai_csv(rows),
            "feature_columns": _XAI_COLS,
            "model_base64": blob,
            "trust_input": True,
        },
    )
    assert envelope["success"] is True
    explanation = envelope["data"]["explanation"]
    contributions = {a["feature"]: a["contribution"] for a in explanation["attributions"]}
    assert all(isinstance(v, float) for v in contributions.values()), contributions
    assert isinstance(explanation["base_value"], float)
    assert isinstance(explanation["prediction"], float)

    # The numbers are checked against the closed form, computed here from the
    # model's own coefficients rather than read back from the envelope: LinearSHAP
    # in log-odds is coef * (x - mean(background)), with the background being the
    # first 50 rows, exactly as main builds it.
    x = rows[0]
    background_mean = rows[:50].mean(axis=0)
    expected = model.coef_[0] * (x - background_mean)
    for name, value, want in zip(_XAI_COLS, contributions.values(), expected):
        assert value == pytest.approx(float(want), abs=1e-12), (name, value, want)
    assert explanation["base_value"] == pytest.approx(
        float(model.intercept_[0] + model.coef_[0] @ background_mean), abs=1e-12
    )
    assert explanation["prediction"] == pytest.approx(
        float(model.intercept_[0] + model.coef_[0] @ x), abs=1e-12
    )
    assert explanation["base_value"] + sum(contributions.values()) == pytest.approx(
        explanation["prediction"], abs=1e-9
    )
    assert explanation["prediction_measured"] is True
    assert explanation["base_value_measured"] is True
    assert explanation["attributions_complete"] is True
    assert explanation["unattributed_features"] == []
    assert explanation["attributions_measured"] is True
    assert "attributions_measured_reason" not in explanation
    assert explanation["local_accuracy_ok"] is True
    assert isinstance(explanation["local_accuracy_residual"], float)
    assert explanation["local_accuracy_residual"] < 1e-6
    assert explanation["background_is_the_explained_instance"] is False
    assert explanation["background_available"] is True
    assert envelope["data"]["method_explicit"] is False
    assert envelope["data"]["routing"]["primary"] == "shap.LinearExplainer"


def test_the_refusal_paths_still_refuse(monkeypatch, capsys, xai_model_blob):
    """The other half of the control: the fix must not turn refusals into
    successes, and must not turn a rendering change into a silent pass."""
    from vfairness.xai import sidecar_cli

    healthy = _xai_csv(np.random.default_rng(1).normal(size=(60, 3)))
    monkeypatch.delenv(sidecar_cli.TRUST_INPUT_ENV, raising=False)
    cases = {
        "row_index out of range": {
            "csv_data": healthy,
            "feature_columns": _XAI_COLS,
            "model_base64": xai_model_blob[0],
            "trust_input": True,
            "row_index": 9999,
        },
        "no csv_data at all": {"model_base64": xai_model_blob[0], "trust_input": True},
        "an unparseable feature value": {
            "csv_data": "a,b,c\nzz,qq,ww\n",
            "feature_columns": _XAI_COLS,
            "model_base64": xai_model_blob[0],
            "trust_input": True,
        },
        "a model blob with no trust opt-in": {
            "csv_data": healthy,
            "feature_columns": _XAI_COLS,
            "model_base64": xai_model_blob[0],
        },
    }
    for name, payload in cases.items():
        _raw, envelope = _run_sidecar(monkeypatch, capsys, payload)
        assert envelope["success"] is False, name
        assert envelope["error"], name
