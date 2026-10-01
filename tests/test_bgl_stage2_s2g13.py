"""Beta Go-Live Stage 2, group s2g13: seven substituted values, at the public entry.

Every test here pins a THIRD STATE (measured / failed / could-not-check) on the
surface a caller actually touches, and every one is paired with a CONTROL on
healthy data, because a fix that makes everything refuse passes any test that
only exercises the degenerate case.

The before-values quoted in each docstring were executed against the code as it
stood on 2026-09-16, not inferred from reading it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness import FairRegressor, NonDeterminismAnalyzer, RAGBiasAnalyzer, TemporalTracker
from vfairness.vision import bias_amplification, ndkl


def _messages(caught) -> str:
    return " || ".join(str(w.message) for w in caught)


# ======================================================================== 1
# non_determinism_analyzer: NonDeterminismAnalyzer.bootstrap_ci
#
# BEFORE: bootstrap_ci([0.6]) -> (0.6, 0.6), a zero-width 95% interval, zero
# warnings. bootstrap_ci(59 real values + 1 NaN) -> (nan, nan), zero warnings,
# byte-identical to the all-NaN case: 59 measurements discarded in silence.
# n == 0 was the only guard.


def test_bootstrap_ci_refuses_a_single_observation_instead_of_claiming_zero_width() -> None:
    analyzer = NonDeterminismAnalyzer("llm")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower, upper = analyzer.bootstrap_ci(np.array([0.6]), random_state=0)

    assert math.isnan(lower) and math.isnan(upper), (
        f"got ({lower}, {upper}); a single observation cannot bound its own uncertainty, "
        f"and a zero-width interval claims certainty no observation supports"
    )
    assert "at least 2 finite values" in _messages(caught), _messages(caught)


def test_bootstrap_ci_keeps_the_59_real_values_when_one_is_not_finite() -> None:
    """The REVERSE defect: real evidence discarded as a mute could-not-check."""
    analyzer = NonDeterminismAnalyzer("llm")
    real = np.random.default_rng(4).normal(0.5, 0.05, 59)
    values = np.concatenate([real, [np.nan]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower, upper = analyzer.bootstrap_ci(values, random_state=0)

    assert math.isfinite(lower) and math.isfinite(upper), (
        f"got ({lower}, {upper}); one non-finite value must not discard 59 measurements"
    )
    assert lower < float(np.mean(real)) < upper
    assert "1 of 60 values are not finite" in _messages(caught), _messages(caught)


def test_bootstrap_ci_refuses_when_every_value_is_non_finite() -> None:
    analyzer = NonDeterminismAnalyzer("llm")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower, upper = analyzer.bootstrap_ci(np.full(5, np.nan), random_state=0)

    assert math.isnan(lower) and math.isnan(upper)
    # Distinguishable from the 59-good-values case ABOVE only by the message,
    # which is the point: both return nan, and only one of them should.
    assert "5 of 5 values are not finite" in _messages(caught), _messages(caught)


def test_bootstrap_ci_control_healthy_series_still_measures_an_interval() -> None:
    """CONTROL. 40 finite runs, above required_runs() for 'llm': a real interval,
    no warning at all."""
    analyzer = NonDeterminismAnalyzer("llm")
    values = np.random.default_rng(7).normal(1.0, 0.5, 40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower, upper = analyzer.bootstrap_ci(values, n_bootstrap=500, random_state=42)

    assert lower < float(np.mean(values)) < upper, (lower, upper)
    assert lower < upper, "a healthy series must not collapse to a zero-width interval"
    assert not caught, _messages(caught)


# ======================================================================== 2
# fair_regressor: FairRegressor.predict_with_sensitive_attr
#
# BEFORE: fitted on groups a/b (offsets {'a': 0.1259, 'b': -0.1259}), called
# with ['a','b','c','c','b','a'] -> per-row delta
# [0.1259, -0.1259, 0.0, 0.0, -0.1259, 0.1259] and ZERO warnings, with no
# attribute recording 'c'. A group whose offset was genuinely MEASURED as 0.0
# produced a byte-identical delta.


def _regression_fixture(n: int = 200):
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (n, 3))
    groups = np.array(["a", "b"] * (n // 2))
    y = X[:, 0] * 2 + (groups == "a") * 3.0 + rng.normal(0, 0.2, n)
    return X, y, groups


def test_fair_regressor_refuses_a_group_fit_never_saw() -> None:
    from sklearn.linear_model import LinearRegression

    X, y, groups = _regression_fixture()
    reg = FairRegressor(LinearRegression(), "mean_parity", tolerance=0.1).fit(
        X, y, sensitive_attr=groups
    )
    # The fixture must actually exercise the branch: real, non-zero offsets.
    assert set(reg.group_offsets_) == {"a", "b"}, reg.group_offsets_
    assert any(abs(v) > 1e-6 for v in reg.group_offsets_.values()), reg.group_offsets_

    with pytest.raises(ValueError, match="no mean-parity offset was fitted"):
        reg.predict_with_sensitive_attr(X[:6], np.array(["a", "b", "c", "c", "b", "a"]))
    assert reg.unseen_groups_ == {"c": 2}, reg.unseen_groups_


def test_fair_regressor_nan_mode_marks_the_untreated_rows_in_the_returned_array() -> None:
    from sklearn.linear_model import LinearRegression

    X, y, groups = _regression_fixture()
    reg = FairRegressor(
        LinearRegression(), "mean_parity", tolerance=0.1, on_unseen_group="nan"
    ).fit(X, y, sensitive_attr=groups)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adjusted = reg.predict_with_sensitive_attr(X[:6], np.array(["a", "b", "c", "c", "b", "a"]))

    unseen_rows = np.array([False, False, True, True, False, False])
    assert np.all(np.isnan(adjusted[unseen_rows])), adjusted
    assert np.all(np.isfinite(adjusted[~unseen_rows])), adjusted
    assert reg.unseen_groups_ == {"c": 2}, reg.unseen_groups_
    assert "no mean-parity offset was fitted" in _messages(caught), _messages(caught)


def test_fair_regressor_control_every_group_seen_is_adjusted_and_silent() -> None:
    """CONTROL. The mitigation still works: all groups seen, real offsets
    applied, nothing recorded as unseen and no warning."""
    from sklearn.linear_model import LinearRegression

    X, y, groups = _regression_fixture()
    reg = FairRegressor(LinearRegression(), "mean_parity", tolerance=0.1).fit(
        X, y, sensitive_attr=groups
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adjusted = reg.predict_with_sensitive_attr(X, groups)

    assert np.all(np.isfinite(adjusted))
    assert reg.unseen_groups_ == {}
    assert not [w for w in caught if "unseen" in str(w.message).lower()], _messages(caught)
    # And it actually mitigates: the adjusted group-mean gap is inside tolerance.
    gap = abs(adjusted[groups == "a"].mean() - adjusted[groups == "b"].mean())
    assert gap <= 0.1 + 1e-9, gap


# ======================================================================== 3
# representation_ndkl: vfairness.vision.ndkl
#
# BEFORE: ndkl([good] + [[]] * 9, ref) -> 0.0506 with warnings=[], byte-identical
# to ndkl([good], ref); the same ten queries with none empty -> 0.6289. Against
# the 0.10 adapters_ranking threshold that is a PASS badge from 10% coverage.

_GOOD_RANKING = ["a", "b"] * 30
_BAD_RANKING = ["a"] * 60
_REF = {"a": 0.5, "b": 0.5}


def test_ndkl_says_how_many_of_the_supplied_rankings_it_measured() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = ndkl([_GOOD_RANKING] + [[]] * 9, _REF)

    assert value.n_rankings_supplied == 10, value.n_rankings_supplied
    assert value.n_rankings_measured == 1, value.n_rankings_measured
    assert value.n_rankings_empty == 9, value.n_rankings_empty
    assert "9 of 10 supplied ranking(s) were empty" in _messages(caught), _messages(caught)
    # The number itself is still the honest mean over what had content.
    assert float(value) == pytest.approx(float(ndkl([_GOOD_RANKING], _REF)))


def test_ndkl_control_ten_full_rankings_measure_ten_and_stay_silent() -> None:
    """CONTROL. Full coverage: nothing dropped, no warning, and the score is the
    one that separates a skewed batch from a balanced one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = ndkl([_GOOD_RANKING] + [_BAD_RANKING] * 9, _REF)

    assert value.n_rankings_supplied == 10
    assert value.n_rankings_measured == 10
    assert value.n_rankings_empty == 0
    assert not caught, _messages(caught)
    assert float(value) > 0.10, value
    # Still a float to every existing caller.
    assert isinstance(value, float)


def test_ndkl_control_a_single_healthy_ranking_is_unchanged() -> None:
    """CONTROL. The one-ranking call that the pulse probe makes is untouched."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = ndkl([_GOOD_RANKING], _REF)
    assert not caught, _messages(caught)
    assert value.n_rankings_measured == 1 and value.n_rankings_empty == 0
    assert 0.0 < float(value) < 0.10


# ======================================================================== 4
# representation_bias_amplification: vfairness.vision.bias_amplification
#
# BEFORE: bias_amplification(['m']*50 + ['x']*50, {'m': 0.5}) ->
# {"available": true, "perGroup": {"m": 0.0, "x": 0.5}, "worstGroup": "x",
#  "maxAmplification": 0.5} with no warning and no unreferencedGroups key, and
# the pulse probe rendered BIA-001 severity 'warn' naming 'x'. The reference
# asserts nothing whatsoever about 'x'.

_GENERATED = ["m"] * 50 + ["x"] * 50


def test_bias_amplification_refuses_a_group_the_reference_never_mentions() -> None:
    out = bias_amplification(_GENERATED, {"m": 0.5})

    assert out["perGroup"]["x"] is None, out
    assert out["unreferencedGroups"] == ["x"], out
    assert out["worstGroup"] != "x", out
    assert out["maxAmplification"] == 0.0, out
    assert "no reference share" in out["note"], out


def test_bias_amplification_separates_a_missing_share_from_a_measured_zero() -> None:
    """The escape the fix has to close: present-and-0.0 is a CLAIM about the
    real world, absent is not, and `.get(key, 0.0)` cannot tell them apart."""
    measured_zero = bias_amplification(_GENERATED, {"m": 1.0, "x": 0.0})
    missing = bias_amplification(_GENERATED, {"m": 1.0})

    assert measured_zero["perGroup"]["x"] == 0.5, measured_zero
    assert measured_zero["unreferencedGroups"] == [], measured_zero
    assert missing["perGroup"]["x"] is None, missing
    assert missing["unreferencedGroups"] == ["x"], missing
    assert measured_zero["perGroup"] != missing["perGroup"]


def test_bias_amplification_an_empty_generated_set_fills_in_nothing() -> None:
    out = bias_amplification([], {"m": 0.5})
    assert out["available"] is False, out
    assert out["maxAmplification"] is None, out
    assert out["worstGroup"] is None, out
    assert out["perGroup"] == {"m": None}, out
    assert "reason" in out


def test_bias_amplification_control_a_fully_referenced_set_is_measured() -> None:
    """CONTROL. Amplification still measured, still graded, still names the
    right worst group when every generated group HAS a reference share."""
    amplifying = bias_amplification(["m"] * 90 + ["x"] * 10, {"m": 0.5, "x": 0.5})
    assert amplifying["available"] is True
    assert amplifying["unreferencedGroups"] == []
    assert amplifying["perGroup"] == {"m": 0.4, "x": -0.4}
    assert amplifying["worstGroup"] == "m"
    assert amplifying["maxAmplification"] == 0.4

    faithful = bias_amplification(_GENERATED, {"m": 0.5, "x": 0.5})
    assert faithful["maxAmplification"] == 0.0
    assert faithful["available"] is True


# ======================================================================== 5
# decompose_mediation
#
# BEFORE, on a DGP with true NDE 1.0, NIE 6.0, proportion mediated 0.857:
#   healthy               -> nde=0.9499 nie=5.8139 prop=0.8596, notes=[]
#   mediator half unknown -> nde=5.4055 nie=1.3583 prop=0.2008, notes=[], warnings=[]
#   constant treatment    -> total=1.8718 nde=0.0 nie=1.8718 prop=1.0, notes=[]
# The middle one INVERTS the conclusion, from "laundered through the mediator"
# to "mostly direct", because unreadable strings were replaced by the column mean.

_GML = (
    'graph[directed 1 node[id "T" label "T"] node[id "M" label "M"] '
    'node[id "Y" label "Y"] edge[source "T" target "M"] '
    'edge[source "M" target "Y"] edge[source "T" target "Y"]]'
)


def _mediation_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    treatment = rng.binomial(1, 0.5, n).astype(float)
    mediator = 3 * treatment + rng.normal(0, 1, n)
    outcome = 1 * treatment + 2 * mediator + rng.normal(0, 1, n)
    return pd.DataFrame({"T": treatment, "M": mediator, "Y": outcome})


def _decompose(frame: pd.DataFrame):
    pytest.importorskip("dowhy")
    from vfairness.operations.causal import decompose_mediation

    result = decompose_mediation(_GML, frame, ["T"], ["Y"], ["M"])
    return result, result.decompositions[0]


def test_mediation_refuses_when_half_the_mediator_cannot_be_read() -> None:
    frame = _mediation_frame()
    frame["M"] = frame["M"].astype(object)
    frame.loc[np.random.default_rng(0).choice(400, 200, replace=False), "M"] = "unknown"
    # The fixture must exercise the coercion path, not the dropna path: these
    # are coded refusals held as STRINGS, so `dropna` never sees them.
    assert frame["M"].isna().sum() == 0
    assert (frame["M"] == "unknown").sum() == 200

    result, dec = _decompose(frame)

    assert dec.natural_direct_effect is None, dec.natural_direct_effect
    assert dec.natural_indirect_effect is None, dec.natural_indirect_effect
    assert dec.proportion_mediated is None, dec.proportion_mediated
    assert any("cannot be read" in note for note in dec.notes), dec.notes
    assert any("DROPPED, not imputed" in note for note in dec.notes), dec.notes
    assert result.warnings, result.warnings
    # The refusal survives serialisation, which is where a reader meets it.
    assert result.to_dict()["decompositions"][0]["proportion_mediated"] is None


def test_mediation_refuses_a_treatment_that_never_varies() -> None:
    frame = _mediation_frame()
    frame["T"] = 1.0
    assert frame["T"].nunique() == 1

    result, dec = _decompose(frame)

    assert dec.natural_direct_effect is None, dec.natural_direct_effect
    assert dec.proportion_mediated is None, dec.proportion_mediated
    # The TOTAL effect is fitted on the same constant treatment and is no more
    # identified than the NDE; refusing only the NDE would move the defect one
    # field over.
    assert dec.total_effect is None, dec.total_effect
    assert any("no contrast" in note for note in dec.notes), dec.notes
    assert result.warnings, result.warnings


def test_mediation_control_healthy_data_recovers_the_true_decomposition() -> None:
    """CONTROL. True NDE 1.0, NIE 6.0, proportion mediated 0.857."""
    result, dec = _decompose(_mediation_frame())

    assert dec.natural_direct_effect == pytest.approx(1.0, abs=0.25), dec.natural_direct_effect
    assert dec.natural_indirect_effect == pytest.approx(6.0, abs=0.5), dec.natural_indirect_effect
    assert dec.proportion_mediated == pytest.approx(0.857, abs=0.05), dec.proportion_mediated
    assert dec.notes == [], dec.notes
    assert result.warnings == [], result.warnings


def test_mediation_control_a_small_unreadable_share_is_still_measured() -> None:
    """CONTROL against over-refusal. 10% unreadable: the decomposition is
    reported on the readable rows, and the note says how many left."""
    frame = _mediation_frame()
    frame["M"] = frame["M"].astype(object)
    frame.loc[np.random.default_rng(1).choice(400, 40, replace=False), "M"] = "unknown"

    _result, dec = _decompose(frame)

    assert dec.proportion_mediated is not None
    assert dec.proportion_mediated == pytest.approx(0.857, abs=0.08), dec.proportion_mediated
    assert any("40 of 400" in note for note in dec.notes), dec.notes


# ======================================================================== 6
# temporal_tracker: TemporalTracker.detect_drift
#
# BEFORE: eight turns of a strictly widening disparity (0.0 -> 0.7) plus one
# all-NaN turn -> detect_drift(0.1) == False, warnings=[]. `abs(nan - 0.0)` is
# nan and `nan > 0.1` is False, so an unmeasurable ENDPOINT returned the verdict
# "no drift beyond the threshold". Every sibling detector on the same tracker
# reported the ramp and carried n_turns_unmeasurable=1.


def _ramp_tracker(n_turns: int = 8) -> TemporalTracker:
    tracker = TemporalTracker()
    for i in range(n_turns):
        gap = i * 0.1
        tracker.record_turn(i, np.full(100, 0.5 + gap / 2), np.full(100, 0.5 - gap / 2))
    return tracker


def test_detect_drift_reads_the_last_measurable_turn_not_a_nan_endpoint() -> None:
    tracker = _ramp_tracker()
    tracker.record_turn(8, np.full(100, np.nan), np.full(100, np.nan))
    # The fixture must actually put a non-finite value at the endpoint.
    _turns, values, n_recorded = tracker._finite_series()
    assert n_recorded == 9 and values.size == 8

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = tracker.detect_drift(0.1)

    assert verdict is True, f"got {verdict!r}; the ramp really does drift 0.0 -> 0.7"
    assert "1 of 9 recorded turns" in _messages(caught), _messages(caught)
    # And it agrees with its own siblings on the same tracker.
    assert tracker.detect_feedback_loop()["has_feedback_loop"] is True


def test_detect_drift_returns_none_when_no_turn_is_measurable() -> None:
    tracker = TemporalTracker()
    for i in range(5):
        tracker.record_turn(i, np.full(100, np.nan), np.full(100, np.nan))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = tracker.detect_drift(0.1)

    assert verdict is None, f"got {verdict!r}; False is the verdict 'no drift'"
    assert "0 of 5" in _messages(caught), _messages(caught)


def test_detect_drift_control_a_healthy_ramp_and_a_healthy_flat_run() -> None:
    """CONTROL, both directions: a real drift is still True and a real absence of
    drift is still False, neither with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _ramp_tracker().detect_drift(0.1) is True
    assert not caught, _messages(caught)

    flat = TemporalTracker()
    rng = np.random.RandomState(42)
    for turn in range(5):
        flat.record_turn(turn, rng.normal(0.5, 0.1, 20), rng.normal(0.5, 0.1, 20))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert flat.detect_drift(0.3) is False
    assert not caught, _messages(caught)


# ======================================================================== 7
# rag_bias_analyzer: RAGBiasAnalyzer.analyze_output
#
# BEFORE: analyze_output(['','',''], ['','','']) -> 0.0 with zero warnings, the
# same 0.0 as ['   ']*3 and ['\n']*3, and verified byte-identical to two
# genuinely identical real answers. full_analysis then shipped
# output_disparity=0.0 with is_output_biased=False. The guard tested
# len(list) == 0, not whether anything was generated.

_REAL_ANSWER = "the policy allows it after twelve months"
_OTHER_ANSWER = "you are not eligible under any circumstances"


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_analyze_output_refuses_when_nothing_was_generated(blank: str) -> None:
    analyzer = RAGBiasAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = analyzer.analyze_output([blank] * 3, [blank] * 3)

    assert math.isnan(value), f"got {value!r}; 0.0 on this scale means identical outputs"
    assert "not measured" in _messages(caught), _messages(caught)


def test_full_analysis_grades_blank_completions_as_could_not_check() -> None:
    analyzer = RAGBiasAnalyzer()
    docs = [{"id": "d1", "text": "policy"}, {"id": "d2", "text": "more"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.full_analysis(["q"], ["q"], docs, docs, [""] * 3, [""] * 3)

    assert math.isnan(result.output_disparity), result.output_disparity
    assert result.is_output_biased is None, result.is_output_biased
    assert "is_output_biased is None" in _messages(caught), _messages(caught)
    assert result.to_dict()["is_output_biased"] is None


def test_analyze_output_keeps_the_finding_when_only_one_group_was_served() -> None:
    """A group that got nothing while the other got answers is a MEASUREMENT,
    the strongest one this metric makes, and must not be refused away."""
    analyzer = RAGBiasAnalyzer()
    assert analyzer.analyze_output([_REAL_ANSWER] * 2, ["", ""]) == 1.0
    mixed = analyzer.analyze_output([_REAL_ANSWER, ""], [_REAL_ANSWER, ""])
    assert mixed == pytest.approx(2.0 / 3.0), mixed


def test_analyze_output_control_real_answers_are_still_scored() -> None:
    """CONTROL. Identical real answers still score 0.0, different real answers
    still score high, and neither warns; the blank run above is no longer
    indistinguishable from the first of these."""
    analyzer = RAGBiasAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agree = analyzer.analyze_output([_REAL_ANSWER] * 3, [_REAL_ANSWER] * 3)
        differ = analyzer.analyze_output([_REAL_ANSWER] * 3, [_OTHER_ANSWER] * 3)

    assert agree == 0.0, agree
    assert differ > 0.5, differ
    assert not caught, _messages(caught)


class _ConstantEmbedder:
    """A stand-in for sentence-transformers that answers the SAME vector for
    every string, including the empty one.

    This is what a real embedder does with blank input: it returns a vector, so
    the cosine of two nothings is 1.0 and the disparity is 0.0, "the two groups
    were answered identically". The embedding branch never reaches the pairwise
    loop, so only the guard ABOVE the branch selection protects it.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.array([1.0, 0.0, 0.0]), (len(texts), 1))


def test_analyze_output_refuses_blank_completions_on_the_embedding_path_too() -> None:
    """The guard sits ABOVE the branch selection, so the embedding path cannot
    inherit the defect the trigram path was fixed for."""
    analyzer = RAGBiasAnalyzer()
    embedder = _ConstantEmbedder()
    # Set BOTH: the constructor silently falls back to trigram when
    # sentence-transformers is absent, and a fixture that only set the embedder
    # would test the trigram path while claiming to test the embedding one.
    analyzer.similarity_method = "embedding"
    analyzer._embedder = embedder
    # The fixture must actually select the embedding branch: prove it on
    # healthy input FIRST, so a green refusal below cannot come from the
    # embedder simply never being consulted.
    assert analyzer.analyze_output([_REAL_ANSWER], [_OTHER_ANSWER]) == pytest.approx(0.0)
    assert embedder.calls, "the embedding branch was never taken; the fixture is wrong"

    embedder.calls.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = analyzer.analyze_output(["", "", ""], ["", "", ""])

    assert math.isnan(value), f"got {value!r}; the embedding path fabricated agreement"
    assert not embedder.calls, "blank completions were sent to the embedder anyway"
    assert "not measured" in _messages(caught), _messages(caught)
