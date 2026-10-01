"""BGL3 batch `_toplevel-1`: do the top-level helpers refuse honestly?

Ten units across five modules (`_names`, `_triage`, `branding`, `status`,
`streaming`), each judged by EXECUTION on an input where the thing it reports
genuinely does not exist, and again on an input where it does. Most of these
bodies are a handful of lines, which is exactly where an answer gets asserted
instead of determined: there is not enough code for the substitution to look
wrong.

Every number quoted in a docstring below was measured on this repo on
2026-09-27, before the fix in the same commit.
"""

from __future__ import annotations

import importlib
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import vfairness
from vfairness import branding
from vfairness._names import name_token_list, resolve_attribute_key, tokens_contain
from vfairness._triage import is_flag, is_measured, unmeasurable_reason
from vfairness.streaming import (
    stream_group_counts,
    streaming_demographic_parity,
    streaming_selection_rates,
)

status_module = importlib.import_module("vfairness.status")


def _run(fn):
    """Call *fn*, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# vfairness._names.name_token_list / tokens_contain
# ---------------------------------------------------------------------------


def test_a_name_nobody_supplied_has_no_tokens():
    """MEASURED BEFORE: `name_token_list(None) == ['none']` and
    `name_token_list(float('nan')) == ['nan']`.

    The `str(name)` coercion turned an absent name into a real, lowercase,
    matchable word out of the module this package nominates as the single
    source of truth for name matching. 'none' is not a word anybody wrote.
    """
    assert name_token_list(None) == []
    assert name_token_list(float("nan")) == []
    assert name_token_list(np.nan) == []
    assert name_token_list(np.float32("nan")) == []


def test_a_keyword_nobody_supplied_never_matches_a_column():
    """MEASURED BEFORE: `tokens_contain('none_reported', None)` was True and
    `tokens_contain('nan_column', float('nan'))` was True.

    A whole-token match, the strongest signal this module hands its callers,
    asserted from a keyword that was never supplied. The proxy detector turns
    such a match into DIRECT, the strongest label in its enum.
    """
    assert tokens_contain("none_reported", None) is False
    assert tokens_contain("nan_column", float("nan")) is False
    assert tokens_contain(None, None) is False
    # An unusable needle is refused without consulting the column at all.
    assert tokens_contain("age_years", "") is False
    assert tokens_contain("age_years", "___") is False


def test_the_string_none_is_still_a_name_somebody_wrote():
    """OVER-CORRECTION CONTROL. Only the missing OBJECTS are refused: a column
    whose label is the four characters "None" still tokenises."""
    assert name_token_list("None") == ["none"]
    assert tokens_contain("none_reported", "none") is True


def test_real_names_still_tokenise_and_still_match():
    """OVER-CORRECTION CONTROL. A guard that refuses everything is as wrong as
    one that answers everything, and the difference is invisible without this.
    """
    assert name_token_list("customerAge") == ["customer", "age"]
    assert name_token_list("AVERAGE_SPEND") == ["average", "spend"]
    assert name_token_list("género") == ["genero"]
    assert tokens_contain("census_tract_id", "census_tract") is True
    assert tokens_contain("average_spend", "age") is False
    assert resolve_attribute_key("sex", ["gender", "race"]) == "gender"
    # And the explicit could-not-resolve is still an explicit None.
    assert resolve_attribute_key(None, ["gender", "race"]) is None
    assert resolve_attribute_key("gender", []) is None


# ---------------------------------------------------------------------------
# vfairness._triage.is_measured / unmeasurable_reason / is_flag
# ---------------------------------------------------------------------------


def test_a_decimal_is_a_measurement():
    """MEASURED BEFORE: `is_measured(Decimal('0.5'))` was False.

    The READINESS-6 numpy-scalar defect over again, in the same direction that
    is hardest to see: a real, finite number reported as a could-not-check, so
    an aggregate drops real evidence and then discloses the loss honestly. The
    stdlib registers Decimal with `numbers.Number` and NOT `numbers.Real`
    (`isinstance(Decimal('0.5'), numbers.Real)` is False), and two other
    coercers in this repo accept Decimal on purpose, so the module written to
    end six disagreeing copies of this rule was the copy that disagreed.
    """
    assert is_measured(Decimal("0.5")) is True
    assert is_measured(Decimal("0")) is True
    assert is_measured(Decimal("-3.25")) is True
    # Accepting the type must not weaken either rule.
    assert is_measured(Decimal("NaN")) is False
    assert is_measured(Decimal("Infinity")) is False
    assert is_measured(Decimal("-Infinity")) is False


def test_unmeasurable_reason_names_a_decimal_nan_instead_of_its_type():
    """MEASURED BEFORE: `unmeasurable_reason(Decimal('0.5'))` was
    'not a number (Decimal)' and `unmeasurable_reason(Decimal('NaN'))` was the
    SAME string.

    Both wrong, and in opposite directions: the first refuses a measurement,
    the second refuses for the wrong reason. The reason is the half an operator
    acts on, because NaN and infinity arrive by different routes.
    """
    assert unmeasurable_reason(Decimal("0.5")) == ""
    assert unmeasurable_reason(Decimal("NaN")) == "NaN: insufficient evidence"
    assert unmeasurable_reason(Decimal("Infinity")) == "infinite: no comparison can grade it"
    # A number that will not read as one is still refused, and by a named
    # reason rather than by an escaping ValueError from float().
    assert "not readable as a number" in unmeasurable_reason(Decimal("sNaN"))


def test_the_triage_pair_never_disagrees_about_the_same_value():
    """`is_measured` False and `unmeasurable_reason` '' would be a hole: the
    caller drops the value and then has nothing to say about it. Checked in
    both directions over every type this module claims to rule on."""
    values = [
        0.0,
        0.5,
        -3.2,
        5,
        np.float32(0.23),
        np.int64(5),
        Fraction(1, 2),
        Decimal("0.5"),
        None,
        float("nan"),
        float("inf"),
        np.float32("nan"),
        True,
        False,
        np.bool_(True),
        "0.5",
        "abc",
        Decimal("NaN"),
        np.array([0.5]),
        1 + 0j,
    ]
    for value in values:
        measured = is_measured(value)
        reason = unmeasurable_reason(value)
        assert measured is not bool(reason), (
            f"{value!r} ({type(value).__name__}): is_measured={measured} "
            f"but unmeasurable_reason={reason!r}"
        )


def test_triage_still_refuses_what_it_always_refused():
    """OVER-CORRECTION CONTROL for the Decimal change: `numbers.Real` plus
    Decimal must not become "anything at all"."""
    for value in [None, float("nan"), float("inf"), float("-inf"), True, np.bool_(False)]:
        assert is_measured(value) is False
    for value in ["0.5", "abc", [1.0], {"a": 1}, np.array([0.5]), 1 + 0j, object()]:
        assert is_measured(value) is False
    assert unmeasurable_reason("0.5") == "not a number (str)"


def test_is_flag_is_true_for_booleans_and_nothing_else():
    """CORRECT, pinned rather than fixed. A flag is never a measurement, and
    `np.bool_` is the half that gets missed: it is not a Python bool, so
    `isinstance(v, bool)` is False for it and a boolean DataFrame column walks
    past the guard and becomes 1.0, which on a drift chart is the maximum.
    """
    assert is_flag(True) is True
    assert is_flag(False) is True
    assert is_flag(np.bool_(True)) is True
    assert is_flag(np.bool_(False)) is True
    # A numeric string is deliberately NOT a flag: the rendering coercers read
    # "0.5" out of JSON as a real measurement and still need to refuse a flag.
    for value in [1, 0, 1.0, "True", "0.5", None, Decimal("1"), np.float32(1.0)]:
        assert is_flag(value) is False, repr(value)


# ---------------------------------------------------------------------------
# vfairness.branding.branding_enabled
# ---------------------------------------------------------------------------


def test_branding_discloses_a_value_it_could_not_read(monkeypatch):
    """CORRECT, pinned rather than fixed. An unrecognised VFAIRNESS_BRANDING
    cannot be resolved to on or off; branding stays ON and the run SAYS so, so
    a typo never silently strips the mark while the user believes it worked.
    """
    branding.set_branding(None)
    monkeypatch.setenv(branding.ENV_VAR, "OFFF")
    out, msgs = _run(branding.branding_enabled)
    assert out is True
    assert any("not a recognised value" in m and "OFFF" in m for m in msgs), msgs

    # The empty string is the same could-not-read, not a falsey value.
    monkeypatch.setenv(branding.ENV_VAR, "")
    out, msgs = _run(branding.branding_enabled)
    assert out is True
    assert any("not a recognised value" in m for m in msgs), msgs


def test_branding_still_answers_the_values_it_does_recognise(monkeypatch):
    """OVER-CORRECTION CONTROL: the switch must still switch."""
    branding.set_branding(None)
    monkeypatch.delenv(branding.ENV_VAR, raising=False)
    assert _run(branding.branding_enabled) == (True, [])
    for raw in ["off", "0", "FALSE", " no ", "none"]:
        monkeypatch.setenv(branding.ENV_VAR, raw)
        assert _run(branding.branding_enabled) == (False, []), raw
    for raw in ["on", "1", "TRUE", "yes"]:
        monkeypatch.setenv(branding.ENV_VAR, raw)
        assert _run(branding.branding_enabled) == (True, []), raw
    try:
        branding.set_branding(False)
        assert branding.branding_enabled() is False
    finally:
        branding.set_branding(None)


# ---------------------------------------------------------------------------
# vfairness.status.require_checked / status
# ---------------------------------------------------------------------------


def test_require_checked_refuses_to_pass_over_nothing():
    """MEASURED BEFORE: `vfairness.require_checked()` returned None, which is
    exactly what it returns when every capability named IS checked.

    The route in is the splatted form the gate is built for,
    `require_checked(*config["capabilities"])`: a config that failed to load or
    a filter that matched nothing makes the tuple empty, and the CI gate goes
    green having verified zero capabilities.
    """
    with pytest.raises(ValueError) as excinfo:
        vfairness.require_checked()
    assert "nothing" in str(excinfo.value)
    with pytest.raises(ValueError):
        vfairness.require_checked(*[])


def test_require_checked_still_passes_a_checked_name_and_still_bites(status_with_every_state):
    """OVER-CORRECTION CONTROL. Refusing the empty call must not turn the gate
    into something that refuses every call, and the gate must still fail on a
    NOT CHECKED name: without both halves this file could pass on a gate that
    only ever raises."""
    ledger = status_module._load()["states_by_unit"]
    checked = next(q for q, s in ledger.items() if s == "CHECKED")
    not_checked = next(q for q, s in ledger.items() if s == "NOT CHECKED")
    assert vfairness.require_checked(checked) is None
    with pytest.raises(RuntimeError) as excinfo:
        vfairness.require_checked(not_checked)
    assert not_checked in str(excinfo.value)


def test_status_refuses_a_name_it_cannot_resolve_and_answers_the_surface():
    """CORRECT, pinned rather than fixed. An unknown name raises instead of
    reading as NOT CHECKED (a typo must not become a finding about the
    library), and `SurfaceStatus.share` raises for a state it does not hold
    rather than returning the 0.0 that a `.get(key, 0)` would have fabricated.
    """
    with pytest.raises(status_module.UnknownCapabilityError):
        vfairness.status("this_name_is_not_a_capability_anywhere")
    with pytest.raises(status_module.UnknownCapabilityError):
        vfairness.status(42)

    surface = vfairness.status()
    assert surface.total > 0
    assert set(surface.counts) <= {"CHECKED", "FIX PENDING", "NOT CHECKED"}
    assert 0.0 < surface.share("CHECKED") <= 1.0
    assert surface.share("checked") == surface.share("CHECKED")
    with pytest.raises(ValueError):
        surface.share("perfect")

    one = vfairness.status("demographic_parity_difference")
    assert one.state in {"CHECKED", "FIX PENDING", "NOT CHECKED"}
    assert one.checked is (one.state == "CHECKED")


# ---------------------------------------------------------------------------
# vfairness.streaming.stream_group_counts
# ---------------------------------------------------------------------------


def test_stream_group_counts_says_when_nothing_qualified():
    """MEASURED BEFORE: 60 rows, groups 'a' and 'b' 30 rows each, every
    prediction NaN, returned `{}` and emitted NOTHING.

    That is byte-identical to `stream_group_counts([], [])`, so the caller
    could not tell a dataset where nothing was measurable from a dataset with
    no groups in it. The whole-array path keeps this number: its
    `handle_missing_values` RETURNS n_excluded. This one returns a dict of
    groups and has no other channel, so the warning IS the channel.
    """
    y_pred = np.full(60, np.nan)
    sensitive = np.array(["a", "b"] * 30, dtype=object)
    counts, msgs = _run(lambda: stream_group_counts(y_pred, sensitive))
    assert counts == {}
    assert any("no group could be counted" in m and "0 of 60" in m for m in msgs), msgs

    # Same collapse through a missing sensitive value rather than a missing
    # prediction, and through the rates wrapper above it.
    rates, msgs = _run(
        lambda: streaming_selection_rates(
            np.array([1, 0] * 30, dtype=float), np.array([None] * 60, dtype=object)
        )
    )
    assert rates == {}
    assert any("no group could be counted" in m for m in msgs), msgs


def test_stream_group_counts_says_how_much_of_the_input_it_kept():
    """MEASURED BEFORE: 60 rows with 30 missing sensitive values returned
    `{'a': (8, 15), 'b': (7, 15)}` in silence.

    The counts are right for what remained. Nothing said half the data had
    left, and a caller reading (8, 15) has no way to find out.
    """
    y_pred = np.array([1, 0] * 30, dtype=float)
    sensitive = np.array((["a"] * 15 + ["b"] * 15 + [None] * 30), dtype=object)
    counts, msgs = _run(lambda: stream_group_counts(y_pred, sensitive))
    assert counts == {"a": (8, 15), "b": (7, 15)}
    assert any("30 of 60 rows" in m and "50.0 percent" in m for m in msgs), msgs


def test_stream_group_counts_is_silent_and_exact_on_a_clean_run():
    """OVER-CORRECTION CONTROL. A run with nothing excluded must warn about
    nothing, and the counts must still be exact and chunk-invariant: the
    streamed number is the reason this module exists."""
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    sensitive = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    for chunk_size in (1, 2, 5, 100_000):
        counts, msgs = _run(
            lambda cs=chunk_size: stream_group_counts(y_pred, sensitive, chunk_size=cs)
        )
        assert counts == {"a": (2, 3), "b": (1, 3)}, chunk_size
        assert msgs == [], (chunk_size, msgs)


# ---------------------------------------------------------------------------
# vfairness.streaming.streaming_demographic_parity
# ---------------------------------------------------------------------------


def test_streaming_dp_discloses_a_group_the_size_gate_dropped():
    """MEASURED BEFORE: 110 rows, 'a' 50 rows selected at 0.5, 'b' 50 rows at
    0.5, 'c' 10 rows at 0.0, default min_group_size=30, returned **0.0 with no
    warning at all**.

    0.0 is perfect parity, the strongest all-clear the metric has, and the
    group it was covering for had been selected at zero percent. The same data
    with min_group_size=10 gives 0.5. The value stays max minus min over the
    eligible groups, because it must keep matching the whole-array result;
    what was missing was saying which group left and how much data with it.
    """
    y_pred = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25 + [0] * 10)
    sensitive = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 10, dtype=object)

    value, msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert value == 0.0
    disclosure = [m for m in msgs if "excluding 1 group" in m]
    assert disclosure, msgs
    assert "min_group_size=30" in disclosure[0]
    assert "'c': 10" in disclosure[0]
    assert "10 of 110 rows" in disclosure[0]
    assert "9.1 percent" in disclosure[0]
    assert "could not check" in disclosure[0]

    # Lowering the gate includes the group and the spread appears, which is the
    # proof that the silent 0.0 was hiding a real gap rather than reporting one.
    included, _ = _run(lambda: streaming_demographic_parity(y_pred, sensitive, min_group_size=10))
    assert included == pytest.approx(0.5)


def test_streaming_dp_and_the_core_metric_disclose_the_same_drop():
    """The streaming docstring claims the value matches the whole-array result
    and the default gate of the `evaluation` metrics. MEASURED BEFORE on
    identical data: the core returned 0.0 AND warned "Excluding 1 group(s)
    below min_group_size=30: {'c': 10}. That leaves 10 of 110 rows (9.1
    percent) out of this metric", while the streaming twin returned 0.0 in
    silence. Two renderings of one measurement must not disagree about how
    much of the data they looked at.
    """
    from vfairness.evaluation.vfairness_metrics.classification import (
        demographic_parity_difference,
    )

    y_pred = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25 + [0] * 10)
    sensitive = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 10, dtype=object)
    y_true = np.zeros_like(y_pred)

    core_value, core_msgs = _run(lambda: demographic_parity_difference(y_true, y_pred, sensitive))
    stream_value, stream_msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert stream_value == pytest.approx(core_value)
    for needle in ("min_group_size=30", "'c': 10", "10 of 110 rows", "9.1 percent"):
        assert any(needle in m for m in core_msgs), (needle, core_msgs)
        assert any(needle in m for m in stream_msgs), (needle, stream_msgs)


def test_streaming_dp_still_fails_closed_when_no_comparison_happened():
    """The refusal this function already had, re-pinned so the new partial-drop
    branch cannot swallow it: one eligible group means no PAIR, so there is
    nothing to compare and the answer is NaN, not 0.0."""
    y_pred = np.array([1] * 50 + [0] * 10)
    sensitive = np.array(["maj"] * 50 + ["min"] * 10, dtype=object)
    value, msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert np.isnan(value)
    assert any("no between-group comparison" in m for m in msgs), msgs


def test_streaming_dp_is_silent_when_every_group_qualified():
    """OVER-CORRECTION CONTROL. Two full groups, nothing dropped: the measured
    gap comes back and no warning is emitted, so the disclosures above mean
    something when they do appear."""
    y_pred = np.array([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40)
    sensitive = np.array(["a"] * 50 + ["b"] * 50, dtype=object)
    value, msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
    assert value == pytest.approx(0.6)
    assert msgs == [], msgs
