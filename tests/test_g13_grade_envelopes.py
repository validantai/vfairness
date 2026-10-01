"""G13 grading pins: streaming, validity aggregation, and the small envelopes.

Fixed on 2026-09-30:

1. ``validity.aggregate.aggregate_validity`` selected its measured subset with
   ``r.value is not None``. A NaN is not None, an infinity is not None and
   ``True`` is not None, so all three were aggregated: ten records carrying
   ``available=True, value=nan`` returned ``available=True``, coverage 1.0, no
   warning, VG-001 mean ``nan`` and VG-005 mean **0.0** against its 0.1 maximum,
   which is a clean pass over a batch that measured nothing.
2. ``streaming.streaming_selection_rates`` drops the denominator, so a rate
   resting on ONE row was published with nothing beside it, while
   ``streaming_demographic_parity`` refused the same data and said why.
3. ``_accumulate_group_counts`` built its group mask as ``(sa_arr == g) & valid``
   over the WHOLE chunk, so ``pd.NA`` in the sensitive column raised
   ``TypeError: boolean value of NA is ambiguous`` -- the one absence door of six
   that crashed, and the one a pandas nullable column produces.
4. ``result.TaskResult.from_envelope`` truthiness-tested the success flag, so
   ``{"success": "false", "error": "the real error"}`` was re-emitted as
   ``success: True`` with that error still attached.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness
from vfairness.result import TaskResult
from vfairness.status import CapabilityStatus
from vfairness.streaming import (
    stream_group_counts,
    streaming_demographic_parity,
    streaming_selection_rates,
)
from vfairness.validity.aggregate import aggregate_validity
from vfairness.validity.groundedness import GroundednessResult
from vfairness.xai.diagnostics.adversarial import (
    AdversarialProbeResult,
    multi_seed_adversarial_probe,
)

NAN = float("nan")


def _run(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(w.message) for w in caught]


# ── 1. aggregate_validity ───────────────────────────────────────────────────


class TestAggregateValidityCountsOnlyRealMeasurements:
    @pytest.mark.parametrize(
        "bad", [NAN, float("inf"), float("-inf"), True], ids=["nan", "inf", "-inf", "True"]
    )
    def test_a_batch_of_nothing_but_unmeasurable_values_fails_closed(self, bad):
        """Before: available=True, coverage 1.0, no warning, and VG-005 mean 0.0
        against a 0.1 maximum -- the strongest all-clear the metric has. For
        value=True the mean was a perfect 1.0."""
        out, _msgs = _run(
            lambda: aggregate_validity(
                [GroundednessResult(value=bad, available=True) for _ in range(10)]
            )
        )

        assert out["available"] is False
        assert out["n_measured"] == 0
        assert "VG-005" not in out and "VG-001" not in out

    def test_one_unmeasurable_record_does_not_destroy_the_other_nine(self):
        """Before: VG-001 mean was `nan` for a batch of nine real 0.95s plus one
        NaN, and coverage still read 1.0 with no warning."""
        rs = [GroundednessResult(value=NAN, available=True)] + [
            GroundednessResult(value=0.95, available=True) for _ in range(9)
        ]

        out, msgs = _run(lambda: aggregate_validity(rs))

        assert out["available"] is True
        assert out["n_measured"] == 9
        assert math.isfinite(out["VG-001"]["mean"])
        assert out["VG-001"]["mean"] == pytest.approx(0.95)
        assert out["coverage"] == pytest.approx(0.9)
        assert any("9 of 10 result(s) were measured" in m for m in msgs)
        assert any("reported available=True while carrying no real" in m for m in msgs)

    def test_the_hallucination_share_is_not_diluted_by_an_unmeasured_record(self):
        """`nan < 1.0` is False, so an unmeasurable record used to count as NOT
        hallucinated and pulled the share DOWN, towards the pass side of the
        0.1 maximum."""
        rs = [GroundednessResult(value=NAN, available=True)] + [
            GroundednessResult(value=0.4, available=True) for _ in range(9)
        ]

        out, _msgs = _run(lambda: aggregate_validity(rs))

        assert out["VG-005"]["mean"] == pytest.approx(1.0), "9 of 9 measured answers"

    def test_control_a_fully_measured_batch_is_unchanged_and_silent(self):
        out, msgs = _run(
            lambda: aggregate_validity(
                [GroundednessResult(value=0.95, available=True) for _ in range(10)]
            )
        )

        assert out["available"] is True
        assert out["n_measured"] == 10
        assert out["coverage"] == 1.0
        assert out["note"] is None
        assert out["VG-001"]["mean"] == pytest.approx(0.95)
        assert msgs == []

    def test_control_a_measured_zero_is_still_a_measurement(self):
        """0.0 is the worst groundedness there is, not an absence."""
        out, _msgs = _run(
            lambda: aggregate_validity(
                [GroundednessResult(value=0.0, available=True) for _ in range(5)]
            )
        )

        assert out["available"] is True
        assert out["n_measured"] == 5
        assert out["VG-001"]["mean"] == 0.0
        assert out["VG-005"]["mean"] == pytest.approx(1.0)


# ── 2 and 3. streaming ──────────────────────────────────────────────────────


class TestStreamingSelectionRatesDisclosesItsDenominators:
    def test_a_rate_resting_on_one_row_is_named(self):
        """Before: {'a': 0.505, 'b': 1.0} in silence, while
        streaming_demographic_parity on the same data returned nan AND warned.
        Two surfaces of one module, opposite readings, one of them mute."""
        y_pred = np.concatenate([np.array([1, 0] * 49 + [1]), np.array([1])]).astype(float)
        sensitive = np.array(["a"] * 99 + ["b"], dtype=object)

        rates, msgs = _run(lambda: streaming_selection_rates(y_pred, sensitive))

        assert rates["b"] == 1.0, "the rate itself is exact and must not be dropped"
        assert any("'b' (n=1)" in m for m in msgs)
        assert any("streaming_selection_rates" in m for m in msgs)
        # And the sibling still refuses the comparison outright.
        dp, dp_msgs = _run(lambda: streaming_demographic_parity(y_pred, sensitive))
        assert math.isnan(dp) and dp_msgs

    def test_the_disclosure_is_emitted_under_this_functions_own_name(self):
        """The exclusion warnings used to arrive from stream_group_counts, so a
        caller was told about a function it had not called."""
        y_pred = np.full(60, np.nan)
        sensitive = np.array(["a", "b"] * 30, dtype=object)

        rates, msgs = _run(lambda: streaming_selection_rates(y_pred, sensitive))

        assert rates == {}
        assert any(
            m.startswith("streaming_selection_rates: no group could be counted") for m in msgs
        )

    def test_control_two_healthy_groups_say_nothing_and_match_the_whole_array(self):
        rng = np.random.RandomState(0)
        y_pred = np.concatenate([rng.binomial(1, 0.6, 2500), rng.binomial(1, 0.35, 2500)])
        sensitive = np.array(["A"] * 2500 + ["B"] * 2500, dtype=object)

        rates, msgs = _run(lambda: streaming_selection_rates(y_pred, sensitive, chunk_size=333))

        whole = {g: float((y_pred[sensitive == g] == 1).mean()) for g in ("A", "B")}
        assert rates == whole, "the streamed result must stay identical to the whole array"
        assert msgs == []

    def test_control_the_floor_is_the_callers_to_set(self):
        y_pred = np.array([1, 0] * 10, dtype=float)
        sensitive = np.array(["a"] * 20, dtype=object)

        _rates, quiet = _run(lambda: streaming_selection_rates(y_pred, sensitive, min_group_size=5))
        _rates, loud = _run(lambda: streaming_selection_rates(y_pred, sensitive, min_group_size=50))
        assert quiet == []
        assert any("n=20" in m for m in loud)


class TestEveryAbsenceDoorInTheSensitiveColumn:
    @pytest.mark.parametrize(
        "absent", [None, np.nan, pd.NA, pd.NaT], ids=["None", "np.nan", "pd.NA", "pd.NaT"]
    )
    def test_a_missing_label_is_excluded_and_never_raises(self, absent):
        """Before, for pd.NA ONLY: TypeError('boolean value of NA is ambiguous')
        out of `(sa_arr == g) & valid`, which compared g against the cells that
        `valid` exists to exclude. pd.NA is what a pandas nullable column gives,
        so this was the ordinary arrival path for a sensitive column with a gap.
        """
        sensitive = np.array(["a"] * 40 + [absent] * 40, dtype=object)
        y_pred = np.array([1, 0] * 40, dtype=float)

        rates, _msgs = _run(lambda: streaming_selection_rates(y_pred, sensitive))
        counts, _m2 = _run(lambda: stream_group_counts(y_pred, sensitive))

        assert rates == {"a": 0.5}
        assert counts == {"a": (20, 40)}

    def test_a_nullable_pandas_column_streams(self):
        sensitive = pd.Series(["a"] * 40 + [None] * 40, dtype="string")
        y_pred = np.array([1, 0] * 40, dtype=float)

        rates, _msgs = _run(
            lambda: streaming_selection_rates(y_pred, sensitive.to_numpy(dtype=object))
        )

        assert rates == {"a": 0.5}

    def test_control_the_counts_are_still_exact_across_chunk_sizes(self):
        y_pred = np.concatenate(
            [np.array([1, 0] * 25, dtype=float), np.array([1] * 5, dtype=float)]
        )
        sensitive = np.array(["a"] * 50 + ["b"] * 5, dtype=object)

        seen = set()
        for chunk in (1, 3, 7, 1000):
            rates, _msgs = _run(
                lambda c=chunk: streaming_selection_rates(y_pred, sensitive, chunk_size=c)
            )
            seen.add(tuple(sorted(rates.items())))
        assert len(seen) == 1
        assert dict(seen.pop()) == {"a": 0.5, "b": 1.0}


# ── 4. TaskResult ───────────────────────────────────────────────────────────


class TestTaskResultAdoptsOnlyABooleanSuccess:
    @pytest.mark.parametrize("raw", ["false", "no", "0", "False", NAN])
    def test_a_non_boolean_success_flag_is_adopted_as_a_failure(self, raw):
        """Before: `bool("false")` is True, so the envelope came back as
        {'success': True, 'error': 'the real error'} -- a failed task published
        as a success, carrying the error that contradicts it."""
        env = {"success": raw, "error": "the real error"}

        out = TaskResult.from_envelope("t", env).to_dict()

        assert out["success"] is False
        assert "not a boolean" in out["error"]
        assert "the real error" in out["error"], "the handler's own reason survives"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (True, True),
            (False, False),
            (np.True_, True),
            (np.False_, False),
            (1, True),
            (0, False),
        ],
    )
    def test_control_a_real_boolean_flag_is_read_as_before(self, raw, expected):
        out = TaskResult.from_envelope("t", {"success": raw, "data": {"x": 1}}).to_dict()
        assert out["success"] is expected

    def test_control_a_missing_success_key_is_still_a_silent_failure(self):
        """The common case, and it must not grow an invented error."""
        out = TaskResult.from_envelope("t", {"data": {"x": 1}}).to_dict()
        assert out["success"] is False
        assert "error" not in out

    def test_control_the_wire_shape_is_unchanged(self):
        assert TaskResult.ok("t", data={"a": 1}).to_dict() == {
            "schema_version": "1.0",
            "task_type": "t",
            "success": True,
            "data": {"a": 1},
        }
        assert TaskResult.fail("t", "boom").to_dict() == {
            "schema_version": "1.0",
            "task_type": "t",
            "success": False,
            "error": "boom",
        }
        assert "data" not in TaskResult.ok("t").to_dict()
        assert "warnings" not in TaskResult.ok("t", data={}, warnings=[]).to_dict()


# ── the units where no defect was found ─────────────────────────────────────


class TestCapabilityStatusFailsClosed:
    """No defect. `checked` is an exact `== "CHECKED"` and `preview` an exact
    `== "preview"`, and BOTH exact tests fail in the safe direction: an
    unrecognised state is not a pass. Matching loosely here would be the
    regression, which is why this is a pin and not a fix."""

    @pytest.mark.parametrize("state", ["checked", "CHECKED ", "Checked", "", None, "PASS"])
    def test_anything_but_the_exact_state_is_not_a_pass(self, state):
        assert CapabilityStatus(name="x", state=state, detail="d").checked is False

    def test_control_the_exact_state_is_a_pass(self):
        assert CapabilityStatus(name="x", state="CHECKED", detail="d").checked is True

    def test_the_two_axes_are_independent(self):
        c = CapabilityStatus(name="x", state="NOT CHECKED", detail="d", scope="preview")
        assert c.checked is False and c.preview is True
        c2 = CapabilityStatus(name="x", state="CHECKED", detail="d", scope="core")
        assert c2.checked is True and c2.preview is False

    def test_the_mutable_defaults_are_not_shared_between_instances(self):
        a = CapabilityStatus(name="a", state="CHECKED", detail="d")
        b = CapabilityStatus(name="b", state="CHECKED", detail="d")
        a.counts["k"] = 1
        assert "k" not in b.counts

    def test_a_real_published_status_reads_its_three_states(self):
        st = vfairness.status("vfairness.streaming.streaming_selection_rates")
        assert st.state in ("CHECKED", "NOT CHECKED")
        assert st.checked is (st.state == "CHECKED")
        assert st.detail


class TestListSkinsHandsBackACopy:
    def test_the_returned_list_is_not_the_live_registry(self):
        from vfairness.rendering.skins import list_skins

        first = list_skins()
        first.append("MUTATED")
        assert "MUTATED" not in list_skins()

    def test_every_listed_skin_is_actually_applicable(self):
        from vfairness.rendering.skins import DEFAULT_SKIN, apply_skin, list_skins

        names = list_skins()
        assert DEFAULT_SKIN in names
        svg = '<svg><rect rx="4" fill="#0aafe3"/></svg>'
        for name in names:
            assert apply_skin(svg, name).startswith("<svg")


class TestPreviewPaletteStatesWhichPaletteItDrew:
    """Not a measurement: colour swatches. It must still not claim to be showing
    a palette it is not, which is what the headline / substitution pair is for
    -- and the disclosure has to be ON the artifact, because a saved file
    carries no warning."""

    @pytest.mark.parametrize("style", ["nope", None, "", 5])
    def test_an_unknown_style_says_so_on_the_figure_itself(self, style):
        fig, msgs = _run(lambda: vfairness.preview_palette(style))

        assert fig is not None
        rendered = fig.to_json() if hasattr(fig, "to_json") else ""
        if rendered:
            assert "is not a known style" in rendered
            assert "MODERN colours" in rendered
        assert any("not a known visualization style" in m for m in msgs)

    @pytest.mark.parametrize("style", ["modern", "dark"])
    def test_control_a_known_style_makes_no_substitution_claim(self, style):
        fig, msgs = _run(lambda: vfairness.preview_palette(style))

        assert fig is not None
        rendered = fig.to_json() if hasattr(fig, "to_json") else ""
        if rendered:
            assert "is not a known style" not in rendered
        assert msgs == []


class TestAdversarialProbeResultCarriesThreeStates:
    """The dataclass itself is not a measurement; what matters is that its
    refusal shape is distinguishable from its clean shape, in the fields a
    consumer reads, and that `flag is None` is never collapsed into False."""

    def test_the_refusal_shape_is_not_a_clean_verdict(self):
        rng = np.random.default_rng(0)
        background = rng.normal(size=(400, 4))
        frozen = np.repeat(background[:1], 400, axis=0)

        def flat(x):
            return np.zeros(len(x))

        clean, _m = _run(
            lambda: multi_seed_adversarial_probe(
                predict_fn=flat, x=background[0], background=background, n_seeds=3
            )
        )
        refused, msgs = _run(
            lambda: multi_seed_adversarial_probe(
                predict_fn=flat, x=frozen[0], background=frozen, n_seeds=3
            )
        )

        assert clean.flag is False and clean.confidence == 1.0
        assert refused.flag is None
        for field in ("confidence", "mean_gap", "fired_fraction"):
            assert math.isnan(getattr(refused, field)), field
        assert "COULD NOT CHECK" in refused.reason
        assert msgs, "a refusal must be audible as well as readable"

    def test_a_bound_no_gap_can_clear_is_refused_not_flagged(self):
        rng = np.random.default_rng(0)
        background = rng.normal(size=(400, 4))

        def flat(x):
            return np.zeros(len(x))

        out, msgs = _run(
            lambda: multi_seed_adversarial_probe(
                predict_fn=flat,
                x=background[0],
                background=background,
                n_seeds=3,
                threshold=-0.5,
            )
        )

        assert out.flag is None
        assert math.isnan(out.confidence)
        assert msgs

    def test_the_three_states_are_all_representable(self):
        for flag in (True, False, None):
            r = AdversarialProbeResult(
                flag=flag, confidence=1.0, mean_gap=0.1, fired_fraction=1.0, reason=""
            )
            assert r.flag is flag
