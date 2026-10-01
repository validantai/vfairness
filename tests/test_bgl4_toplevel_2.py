"""BGL4 audit of batch `A-_toplevel-2`: the overturn demonstration.

BGL3 graded ``vfairness.streaming.streaming_demographic_parity`` PROVEN on the
strength of a new partial-drop disclosure for the ``min_group_size`` gate. The
gate is only ONE of the two routes by which a group can leave the comparison.
The other is missing data: a group whose every row carries a NaN prediction
never reaches ``counts`` at all, so it can never appear in ``dropped_sizes``
(that dict is built by iterating ``counts.items()``), and the function returns
0.0, perfect parity, with nothing said about the group at all.

The whole-array path this module promises to agree with DOES say it, by name:
``_validation.validate_inputs`` tracks the group set before and after exclusion
and warns "Group(s) c lost every row and are absent from every comparison
computed from this data", a disclosure added with the measured note that
dropping one never-selected group's rows moved a reported parity gap from 0.475
to 0.050, "from the starkest possible finding to a pass".

Measured on this repo on 2026-09-27. FIXED THE SAME DAY (BGL5): the accumulation
now reports the group set the DATA held, so both streaming surfaces name a group
that lost every row, with the core's own sentence. This file was the evidence for
the audit row and is now the pin for the corrected behaviour: the xfail markers had
to go, because under ``xfail_strict`` an XPASS is a failure. Each test keeps its
subject and its docstring; only the expected answers are the corrected ones. The
fix, its measured before and after, and its sabotage record are in
src/vfairness/streaming.py, src/vfairness/branding.py and
tests/test_bgl5_toplevel_and_net.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
)
from vfairness.streaming import stream_group_counts, streaming_demographic_parity


def _run(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(w.message) for w in caught]


def _fixture():
    """140 rows, three groups. 'a' and 'b' are 50 rows each selected at 0.5.
    'c' is 40 rows whose predictions are ALL NaN, so its selection rate cannot
    be measured at all: a classifier that abstained on one subpopulation.
    """
    y_pred = np.array([1.0] * 25 + [0.0] * 25 + [1.0] * 25 + [0.0] * 25 + [np.nan] * 40)
    sensitive = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 40, dtype=object)
    return y_pred, sensitive


def test_measured_now_streaming_dp_returns_perfect_parity_for_a_vanished_group():
    """The defect, as measured, CONVERTED TO A PIN by the BGL5 fix.

    The VALUE is unchanged and deliberately so: 0.0 is what the whole-array twin
    returns on this data too, and this module's contract is that the two agree.
    What was wrong was everything around it. Before the fix this test recorded that
    group 'c' was named nowhere and that no could-not-check was reported; both
    assertions are now inverted, and the row-count disclosure they sat beside is
    unchanged.
    """
    y_pred, sensitive = _fixture()
    value, msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))

    # 0.0 is perfect parity, the strongest all-clear the metric has, so the group
    # absent from it has to be named beside it.
    assert value == 0.0
    assert [m for m in msgs if "'c'" in m], msgs
    assert [m for m in msgs if "could not check" in m], msgs
    # The row count is still said, as it was before.
    assert any("were excluded for a missing prediction" in m for m in msgs), msgs

    # The group is still absent from the counts, which is WHY the size-gate
    # disclosure could never see it: dropped_sizes iterates counts.items(), and
    # 'c' is not a key there. The fix reads the group set the data held instead,
    # so stream_group_counts names it too.
    counts, count_msgs = _run(lambda: stream_group_counts(y_pred, sensitive))
    assert set(counts) == {"a", "b"}
    assert any("'c'" in m and "lost every row" in m for m in count_msgs), count_msgs


def test_the_core_metric_and_the_streaming_twin_now_name_the_same_group():
    """The same input through the whole-array path the module promises to match.

    Was ``test_measured_now_the_core_metric_names_the_group_the_streaming_twin does
    not``: the two assertions about the streaming twin's silence are inverted by the
    BGL5 fix, which gave it the core's sentence verbatim. The subject, that two
    renderings of one measurement must not disagree about how much of the data they
    looked at, is unchanged.
    """
    y_pred, sensitive = _fixture()
    y_true = np.zeros(140)

    core_value, core_msgs = _run(lambda: demographic_parity_difference(y_true, y_pred, sensitive))
    stream_value, stream_msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))

    assert core_value == pytest.approx(stream_value) == 0.0
    needle = "lost every row and are absent from every comparison"
    assert any(needle in m for m in core_msgs), core_msgs
    assert any(needle in m for m in stream_msgs), stream_msgs
    assert any("Group(s) c" in m for m in core_msgs), core_msgs
    assert any("'c'" in m and needle in m for m in stream_msgs), stream_msgs


# WAS xfail until the BGL5 fix. The recorded reason: "BGL4 audit overturn, batch
# A-_toplevel-2: streaming_demographic_parity returns 0.0 for a comparison a whole
# group is absent from, when the group left through missing data rather than through
# the min_group_size gate. The whole-array path discloses it by name. Remove this
# marker with the fix."
def test_streaming_dp_should_disclose_a_group_that_lost_every_row():
    """What the PROVEN grade claims is already true."""
    y_pred, sensitive = _fixture()
    _, msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert any("c" in m and ("lost every row" in m or "absent from" in m) for m in msgs), msgs


# ---------------------------------------------------------------------------
# vfairness.branding.set_branding, graded SEMI-PROVEN by a probe
# ---------------------------------------------------------------------------


@pytest.fixture()
def _clean_branding(monkeypatch):
    from vfairness import branding

    monkeypatch.delenv(branding.ENV_VAR, raising=False)
    branding.set_branding(None)
    try:
        yield branding
    finally:
        branding.set_branding(None)


def test_set_branding_no_longer_resolves_a_value_it_cannot_read(_clean_branding):
    """BGL4 audit finding, batch A-_toplevel-2, CONVERTED TO A PIN by the BGL5 fix.

    ``branding_enabled`` is PROVEN because an unrecognised VFAIRNESS_BRANDING
    value is DISCLOSED rather than resolved. ``set_branding`` is step 1 of the
    same documented resolution order and resolved the identical vocabulary with
    a bare ``bool()``, silently, in the OPPOSITE direction: every one of the
    five tokens the module docstring lists as disabling branding turned branding
    ON when handed to the setter, and the empty string, which the environment
    route refuses as unreadable, turned it OFF.

    The subject and the sweep are unchanged. The expected answers are the ones the
    module documents: the two routes agree on every token, and the unreadable value
    is disclosed instead of resolved.
    """
    branding = _clean_branding

    # The module docstring's own disabling vocabulary, through the setter.
    for token in ["off", "0", "false", "no", "none", "OFF"]:
        branding.set_branding(None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            branding.set_branding(token)
            enabled = branding.branding_enabled()
        assert enabled is False, token
        assert [str(w.message) for w in caught] == [], token

        # Same token through the environment: refused or honoured, never silent
        # and never the opposite answer.
        branding.set_branding(None)
        os_env_answer = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import os

            os.environ[branding.ENV_VAR] = token
            os_env_answer = branding.branding_enabled()
            del os.environ[branding.ENV_VAR]
        assert os_env_answer is False, token
        assert os_env_answer is enabled, token
        assert [str(w.message) for w in caught] == [], token

    # And the unreadable value the environment route warns about is now disclosed
    # by the setter too, instead of stripping the mark without a word.
    branding.set_branding(None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        branding.set_branding("")
        enabled = branding.branding_enabled()
    assert enabled is True
    assert [m for m in (str(w.message) for w in caught) if "not a value this switch can read" in m]


# WAS xfail until the BGL5 fix. The recorded reason: "BGL4 audit overturn, batch
# A-_toplevel-2: set_branding coerces any object with bool() and discloses nothing,
# so the five tokens its own module docstring lists as disabling branding silently
# ENABLE it, and '' silently disables it. Remove this marker with the fix."
def test_set_branding_should_not_silently_invert_its_documented_vocabulary(_clean_branding):
    branding = _clean_branding
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        branding.set_branding("off")
        enabled = branding.branding_enabled()
    msgs = [str(w.message) for w in caught]
    assert enabled is False or msgs, (enabled, msgs)
