"""BGL-5 pins for batch A-evaluation-2: the five OVERTURNED rows, closed.

An independent audit re-attacked five grades in this batch and overturned every
one of them. Each defect below was reproduced by execution first, then fixed at
its root, and this file is the evidence that the fix is real: for every one there
is a REFUSAL pin (the unmeasurable input is now refused or disclosed) and an
OVER-CORRECTION CONTROL asserting that healthy input still gets its exact
measured number, so a fix that simply refuses everything fails here too.

The five, with the measured before and after:

  D1  ``bootstrap_over_index(20, mean, groups=['a'] * 5)`` returned
      point_estimate 9.5 with the interval [0.845, 3.155] (rows 0..4 only) and
      sample_size 20, in silence: an interval that does not contain its own
      point estimate. Now InvalidDataError, at this entry point and at
      ``stratified_bootstrap_ci`` itself.
  D2  ``bootstrap_over_index(2, mean, groups=['a', 'b'])`` returned
      lower_bound == upper_bound == 0.5 and standard_error 0.0, in silence: a
      zero-width confidence interval asserts the statistic is known exactly,
      and every replicate was identical because a stratum of one row can only
      resample to itself. Now nan bounds and a UserWarning.
  D3  ``minimum_detectable_effect(2, 2)`` returned a finite 1.0 for an
      arithmetic floor of 1.4008, in silence, because of ``min(mde, 1.0)``. A
      difference of two proportions cannot exceed 1.0, so no attainable floor
      exists. Now nan and a UserWarning.
  D4  ``power_warning(2, 100)`` returned "Very low statistical power (n=2).
      This test can only detect disparities larger than 100%", while n=0 and
      n=1 said "COULD NOT BE ASSESSED" for the same impossibility. Now all
      three agree, and power_warning carries its own guard against printing a
      floor no disparity can reach.
  D5  ``validate_probabilities(np.array([0.5, nan, 0.7], dtype=object))``
      accepted the column (the NaN guard was gated on a floating dtype), and
      ``expected_calibration_error`` then published bin counts
      [22, 20, 18, 22, 13, 14, 20, 20, 10, 41] with an accuracy of 0.4878 over
      the 20 rows that had no probability. Now both refuse.
  D6  ``selection_rate_disparity_matrix`` on 50 A at rate 0.8 plus 2 B at rate
      0.0 published min_ratio 0.0 for the pair ('B', 'A') in silence, from a
      two-row group its own output grades tier 'invalid', against the rule its
      own comment states. Now the headline is nan with headline_basis
      'no_interpretable_pair', a warning, and the old values disclosed under
      headline_over_all_groups.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.evaluation.vfairness_metrics._statistics as S
import vfairness.evaluation.vfairness_metrics.classification as C
from vfairness.evaluation.vfairness_metrics._statistics import (
    bootstrap_over_index,
    minimum_detectable_effect,
    power_warning,
    stratified_bootstrap_ci,
)
from vfairness.evaluation.vfairness_metrics._validation import validate_probabilities
from vfairness.exceptions import InvalidDataError


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


# ---------------------------------------------------------------------------
# D1  the strata must describe the rows being resampled
# ---------------------------------------------------------------------------


def test_bootstrap_over_index_refuses_groups_that_do_not_describe_the_n_rows():
    """A groups array shorter than n can only index the rows it covers.

    Before: point_estimate 9.5 over all 20 rows, interval [0.845, 3.155] over
    rows 0..4, sample_size 20, no warning. An interval that does not contain the
    estimate it is published beside is not a wider interval, it is a different
    measurement. Every other entry point in the library refuses this
    (check_consistent_length, selection_rate_disparity_matrix, validate_inputs).
    """
    data = np.arange(20, dtype=float)
    with pytest.raises(InvalidDataError) as excinfo:
        bootstrap_over_index(
            20,
            lambda idx: float(np.mean(data[idx])),
            groups=np.array(["a"] * 5),
            n_bootstrap=50,
            random_state=3,
        )
    message = str(excinfo.value)
    assert "5" in message and "20" in message, message
    # WHICH GUARD ANSWERED MATTERS. stratified_bootstrap_ci carries the same
    # check, so a pin that only asserts "it raised" stays green when THIS
    # function's check is removed: the call would simply forward and be refused
    # one level down, with a message naming ``data`` instead of ``n``. Pin the
    # arrival, so removing the check here is visible.
    assert "bootstrap_over_index:" in message, message
    assert "n=20" in message, message


def test_the_stratified_entry_point_refuses_the_same_mismatch_directly():
    """The guard is not only above the dispatch: a direct caller arrives too.

    ``stratified_bootstrap_ci`` is public and takes (data, groups) itself, so a
    check only in ``bootstrap_over_index`` would leave the same truncation
    reachable one call lower down.
    """
    with pytest.raises(InvalidDataError):
        stratified_bootstrap_ci(
            np.arange(20, dtype=float),
            np.array(["a"] * 5),
            lambda d, g: float(np.mean(d)),
            n_bootstrap=50,
            random_state=3,
        )


# ---------------------------------------------------------------------------
# D2  one row per stratum is not a resample
# ---------------------------------------------------------------------------


def test_bootstrap_over_index_refuses_a_zero_width_stratified_interval():
    """Every stratum holding one row means every replicate is the original.

    Before: lower_bound == upper_bound == 0.5 and standard_error 0.0, warnings
    []. The ``len(data) < 2`` guard cannot see this, because the sample as a
    whole is big enough; only the per-stratum sizes show it.
    """
    data = np.arange(2, dtype=float)
    result, messages = _caught(
        lambda: bootstrap_over_index(
            2,
            lambda idx: float(np.mean(data[idx])),
            groups=np.array(["a", "b"]),
            n_bootstrap=50,
            random_state=3,
        )
    )
    assert math.isnan(result.lower_bound)
    assert math.isnan(result.upper_bound)
    assert result.standard_error is None
    assert result.n_bootstrap == 0
    # The point estimate WAS measured and is kept: this is could-not-check on
    # the interval, not a refusal of the statistic.
    assert result.point_estimate == pytest.approx(0.5)
    assert any("single row" in m and "identical" in m for m in messages), messages


def test_control_a_stratified_bootstrap_still_measures_a_real_interval():
    """OVER-CORRECTION CONTROL, with the actual numbers.

    Two strata of ten rows each over ``np.arange(20)``: the interval must be
    real, non-degenerate, and must bracket the point estimate. A fix that nan'd
    every stratified interval passes the refusal pins above and fails here.
    """
    data = np.arange(20, dtype=float)
    groups = np.array(["a"] * 10 + ["b"] * 10)
    result, messages = _caught(
        lambda: bootstrap_over_index(
            20,
            lambda idx: float(np.mean(data[idx])),
            groups=groups,
            n_bootstrap=200,
            random_state=7,
        )
    )
    assert result.point_estimate == pytest.approx(9.5)
    assert result.lower_bound == pytest.approx(8.5475, abs=1e-4)
    assert result.upper_bound == pytest.approx(10.75125, abs=1e-4)
    assert result.standard_error == pytest.approx(0.5374, abs=1e-4)
    assert result.lower_bound < result.point_estimate < result.upper_bound
    assert result.n_bootstrap == 200
    assert result.sample_size == 20
    assert messages == [], messages


def test_control_a_single_row_stratum_beside_a_real_one_is_still_measured():
    """OVER-CORRECTION CONTROL for the per-stratum guard specifically.

    The guard fires only when NO stratum has two rows. With 20 rows in 'a' and
    one in 'b' the resample does vary, so the interval is real and must survive:
    point 10.0 inside [7.5226, 12.0964], standard_error 1.2369.
    """
    data = np.arange(21, dtype=float)
    groups = np.array(["a"] * 20 + ["b"])
    result, messages = _caught(
        lambda: bootstrap_over_index(
            21,
            lambda idx: float(np.mean(data[idx])),
            groups=groups,
            n_bootstrap=200,
            random_state=3,
        )
    )
    assert result.point_estimate == pytest.approx(10.0)
    assert result.lower_bound == pytest.approx(7.52262, abs=1e-4)
    assert result.upper_bound == pytest.approx(12.09643, abs=1e-4)
    assert result.standard_error == pytest.approx(1.23688, abs=1e-4)
    assert messages == [], messages


# ---------------------------------------------------------------------------
# D3  minimum_detectable_effect: no floor clamped to the edge of the range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n1,n2", [(2, 2), (2, 3), (3, 3), (2, 5), (2, 100)])
def test_no_minimum_detectable_effect_is_invented_when_no_attainable_floor_exists(n1, n2):
    """The arithmetic floor exceeds 1.0, so the design can detect nothing.

    Before, each of these returned a finite 1.0 with no warning, from
    ``min(mde, 1.0)``: the raw floors were 1.4008, 1.2787, 1.1437, 1.1720 and
    1.0004. A finite 1.0 is the one value that reads as "a 100-point gap is
    detectable" AND passes every ``np.isfinite`` guard downstream.
    """
    z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    raw = (z_alpha + z_power) * math.sqrt(0.25 * (1 / n1 + 1 / n2))
    assert raw > 1.0, "fixture no longer selects a design with no attainable floor"

    value, messages = _caught(lambda: minimum_detectable_effect(n1, n2))
    assert math.isnan(value), f"({n1}, {n2}) returned {value!r}"
    assert value != 1.0
    assert any("cannot exceed" in m and "nan" in m for m in messages), messages


def test_control_a_real_design_still_gets_its_exact_floor():
    """OVER-CORRECTION CONTROL, with the actual numbers and a shape check.

    A clamped constant could not be monotone in n, so the ordering is asserted
    as well as the three exact values.
    """
    a, msgs_a = _caught(lambda: minimum_detectable_effect(3, 100))
    b, msgs_b = _caught(lambda: minimum_detectable_effect(30, 100))
    c, msgs_c = _caught(lambda: minimum_detectable_effect(100, 1000))
    assert a == pytest.approx(0.8207895653160011)
    assert b == pytest.approx(0.2915982346576338)
    assert c == pytest.approx(0.14691636828297927)
    assert a > b > c, "the floor must fall as the arms grow"
    assert msgs_a == [] and msgs_b == [] and msgs_c == []


# ---------------------------------------------------------------------------
# D4  power_warning: the three smallest subgroups must not disagree
# ---------------------------------------------------------------------------


def test_power_warning_treats_a_design_that_can_detect_nothing_the_same_at_0_1_and_2():
    """n=2 used to be graded into the low-power band, n=0 and n=1 were refused.

    Before: power_warning(2, 100) -> "Very low statistical power (n=2). This test
    can only detect disparities larger than 100%", while power_warning(1, 100)
    and power_warning(0, 100) both said "COULD NOT BE ASSESSED". A floor "larger
    than 100%" is unattainable for a difference of proportions, so all three are
    the same impossibility and must read the same way.
    """
    verdicts = {n: power_warning(n, 100) for n in (0, 1, 2)}
    for n, verdict in verdicts.items():
        assert verdict is not None, n
        assert "COULD NOT BE ASSESSED" in verdict, (n, verdict)
        assert "larger than 100%" not in verdict, (n, verdict)
        assert "Very low statistical power" not in verdict, (n, verdict)


@pytest.mark.parametrize("fake_mde", [1.0, 1.4008])
def test_power_warning_never_prints_a_floor_no_disparity_can_reach(monkeypatch, fake_mde):
    """The guard inside power_warning, independent of its producer.

    The clamp is gone from ``minimum_detectable_effect``, so this branch is not
    reachable through it today. It exists because this function, not its
    producer, owns the sentence a reader sees: if any future producer hands it a
    floor at or above 1.0 it must not print "can only detect disparities larger
    than 100%". Driven here by replacing the producer.
    """
    monkeypatch.setattr(S, "minimum_detectable_effect", lambda *a, **k: fake_mde)
    verdict = S.power_warning(50, 1000)
    assert verdict is not None
    assert "larger than 100%" not in verdict, verdict
    assert "NO detectable disparity" in verdict, verdict
    assert "total separation" in verdict, verdict


def test_power_warning_does_not_round_a_real_floor_up_to_an_unattainable_100_percent():
    """0.996 is a measurement, and "{:.0%}" renders it as the unattainable 100%.

    The near-boundary floor keeps a decimal place instead, so the sentence a
    reader gets is a floor a disparity could actually reach.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(S, "minimum_detectable_effect", lambda *a, **k: 0.996)
        verdict = S.power_warning(50, 1000)
    finally:
        monkeypatch.undo()
    assert "99.6%" in verdict, verdict
    assert "100%" not in verdict, verdict


def test_control_power_warning_still_grades_the_designs_it_can_and_still_clears_one():
    """OVER-CORRECTION CONTROL, with the actual floors in the sentences.

    n=3 in 100 has a real floor of 0.8208 and must still be graded, not refused;
    a 300-in-1000 subgroup must still come back None (adequate power), which is
    the state a fix that refused everything would have destroyed.
    """
    three = power_warning(3, 100)
    assert "Very low statistical power (n=3)" in three
    assert "larger than 82%" in three
    assert "COULD NOT BE ASSESSED" not in three

    # 31%, not the 29% this line asserted until 2026-09-30. The second arm of the
    # two-proportion test is the COMPLEMENT (70 rows), not the whole 100, which
    # counted the subgroup on both sides; 29% was a floor computed over a population
    # that does not exist. The subject of the assertion is unchanged: a real design
    # is still GRADED with a floor and not refused.
    thirty = power_warning(30, 100)
    assert "larger than 31%" in thirty, thirty
    assert "COULD NOT BE ASSESSED" not in thirty

    assert power_warning(300, 1000) is None
    assert "Very low statistical power" in power_warning(50, 1000)


# ---------------------------------------------------------------------------
# D5  validate_probabilities: the missing-value guard is not dtype-gated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,probs",
    [
        ("float64 nan", np.array([0.5, np.nan, 0.7])),
        ("object nan", np.array([0.5, np.nan, 0.7], dtype=object)),
        ("object None", np.array([0.5, None, 0.7], dtype=object)),
        ("object Series", pd.Series([0.5, np.nan, 0.7], dtype=object)),
        ("nullable Float64", pd.Series([0.5, None, 0.7], dtype="Float64")),
    ],
)
def test_validate_probabilities_refuses_a_missing_probability_in_every_dtype(label, probs):
    """Before, only "float64 nan" and "nullable Float64" were refused.

    ``np.issubdtype(probs.dtype, np.floating)`` is False for an object column, so
    the guard was skipped and the range check below it cannot catch NaN (every
    comparison with NaN is False). "object None" did not even reach a refusal: it
    raised TypeError out of ``probs < 0``.
    """
    with pytest.raises(InvalidDataError) as excinfo:
        validate_probabilities(np.asarray(probs), "y_prob")
    assert "1 NaN value(s)" in str(excinfo.value), (label, str(excinfo.value))


def test_the_calibration_entry_point_refuses_an_object_column_with_no_probability():
    """The consumer half: the refusal has to reach the surface a person reads.

    Before, with 200 rows of which 20 had no probability, the float column was
    refused and the IDENTICAL values in an object column returned a result whose
    bin table swept all 20 unscored rows into the last bin (counts
    [22, 20, 18, 22, 13, 14, 20, 20, 10, 41]) and reported accuracy 0.4878 over
    it, disclosing nothing but two numpy RuntimeWarnings.
    """
    from vfairness import expected_calibration_error

    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, n)
    probs = np.clip(rng.random(n), 0.01, 0.99)
    probs[:20] = np.nan

    with pytest.raises(InvalidDataError) as float_refusal:
        expected_calibration_error(y_true, probs)
    assert "20 NaN value(s)" in str(float_refusal.value)

    with pytest.raises(InvalidDataError) as object_refusal:
        expected_calibration_error(y_true, pd.Series(list(probs), dtype=object))
    assert "20 NaN value(s)" in str(object_refusal.value)


def test_control_real_probabilities_are_still_accepted_in_every_dtype():
    """OVER-CORRECTION CONTROL, with the actual calibration number.

    ``pd.isna`` is all-False for an int array and for a clean object column, so
    the wider guard must not refuse anything that carries a value: 0/1 integer
    labels, an object column of floats and an empty array all pass, and the
    public calibration entry returns its real ECE for both dtypes.
    """
    from vfairness import expected_calibration_error

    for probs in (
        np.array([0.5, 0.6, 0.7]),
        np.array([0.5, 0.6, 0.7], dtype=object),
        np.array([0, 1, 1]),
        np.array([]),
    ):
        accepted, messages = _caught(lambda p=probs: validate_probabilities(p, "y_prob"))
        assert accepted is None
        assert messages == [], messages

    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, n)
    probs = np.clip(rng.random(n), 0.01, 0.99)
    assert expected_calibration_error(y_true, probs).overall_value == pytest.approx(
        0.24362046606074023
    )
    as_object = pd.Series(list(probs), dtype=object)
    assert expected_calibration_error(y_true, as_object).overall_value == pytest.approx(
        0.24362046606074023
    )


# ---------------------------------------------------------------------------
# D6  selection_rate_disparity_matrix: the headline needs an interpretable pair
# ---------------------------------------------------------------------------


def test_the_headline_disparity_is_not_taken_from_an_uninterpretable_group():
    """50 A at rate 0.8 plus 2 B at rate 0.0, ordinary fairness-audit data.

    Before: min_ratio 0.0 for the pair ('B', 'A') and max_difference 0.8, in
    silence, because ``pool = interp if len(interp) >= 2 else names`` put every
    group back in exactly when the rule "a 4-person group's rate is noise and
    must not define the headline disparity" was about to bite. 0.0 is the
    starkest adverse-impact reading this statistic has.
    """
    y_pred = np.array([1] * 40 + [0] * 10 + [0] * 2)
    sens = np.array(["A"] * 50 + ["B"] * 2)
    out, messages = _caught(lambda: C.selection_rate_disparity_matrix(y_pred, sens))

    assert out["rates"]["B"]["n"] == 2
    assert out["rates"]["B"]["tier"] == "invalid"
    assert out["rates"]["B"]["interpretable"] is False

    assert math.isnan(out["min_ratio"]), out["min_ratio"]
    assert out["min_ratio_pair"] is None
    assert math.isnan(out["max_difference"]), out["max_difference"]
    assert out["headline_basis"] == "no_interpretable_pair"
    assert any("fewer than two groups are interpretable" in m for m in messages), messages
    assert any("n=2, tier=invalid" in m for m in messages), messages

    # Nothing measured is lost: the value that used to be the headline is
    # disclosed under its own name, and the per-pair matrix still carries it.
    disclosed = out["headline_over_all_groups"]
    assert disclosed["min_ratio"] == pytest.approx(0.0)
    assert disclosed["min_ratio_pair"] == ("B", "A")
    assert disclosed["max_difference"] == pytest.approx(0.8)
    assert disclosed["uninterpretable_groups"] == ["B"]
    assert out["ratio_matrix"]["B"]["A"] == pytest.approx(0.0)
    assert out["difference_matrix"]["A"]["B"] == pytest.approx(0.8)


def test_two_groups_that_are_both_uninterpretable_also_refuse_the_headline():
    """2 A plus 2 B: before, min_ratio 0.0 and max_difference 1.0, in silence."""
    out, messages = _caught(
        lambda: C.selection_rate_disparity_matrix(
            np.array([1, 1, 0, 0]), np.array(["A", "A", "B", "B"])
        )
    )
    assert math.isnan(out["min_ratio"])
    assert math.isnan(out["max_difference"])
    assert out["headline_basis"] == "no_interpretable_pair"
    assert out["headline_over_all_groups"]["min_ratio"] == pytest.approx(0.0)
    assert sorted(out["headline_over_all_groups"]["uninterpretable_groups"]) == ["A", "B"]
    assert any("fewer than two groups are interpretable" in m for m in messages), messages


def test_control_two_interpretable_groups_keep_their_measured_headline_in_silence():
    """OVER-CORRECTION CONTROL, with the actual numbers.

    50 A at 0.8 against 20 B at 0.4: both groups clear the interpretable bound
    (n >= 10), so the headline is a real measurement and the call must stay
    silent. A fix that nan'd the headline whenever any group was small, or that
    warned on every call, fails here.
    """
    y_pred = np.array([1] * 40 + [0] * 10 + [1] * 8 + [0] * 12)
    sens = np.array(["A"] * 50 + ["B"] * 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = C.selection_rate_disparity_matrix(y_pred, sens)
    assert out["rates"]["A"]["rate"] == pytest.approx(0.8)
    assert out["rates"]["B"]["rate"] == pytest.approx(0.4)
    assert out["min_ratio"] == pytest.approx(0.5)
    assert out["min_ratio_pair"] == ("B", "A")
    assert out["max_difference"] == pytest.approx(0.4)
    assert out["max_difference_pair"] == ("A", "B")
    assert out["headline_basis"] == "interpretable_groups"
    assert out["headline_over_all_groups"] is None


def test_control_the_nobody_selected_refusal_is_untouched():
    """The 0/0 fix this row was originally graded for must still hold.

    50 A + 50 B with every prediction 0: both groups ARE interpretable, so the
    headline pool is unchanged and the refusal comes from the earlier fix (every
    ratio is 0/0), with max_difference still the measured 0.0.
    """
    out, messages = _caught(
        lambda: C.selection_rate_disparity_matrix(
            np.zeros(100, dtype=int), np.array(["A"] * 50 + ["B"] * 50)
        )
    )
    assert math.isnan(out["min_ratio"])
    assert out["max_difference"] == pytest.approx(0.0)
    assert out["headline_basis"] == "interpretable_groups"
    assert out["headline_over_all_groups"] is None
    assert any("no group has a positive selection rate" in m for m in messages), messages


# ---------------------------------------------------------------------------
# D7  _warn_dropped_groups asked a question it would not read as a finding
# ---------------------------------------------------------------------------
#
# Handed to this batch by the coordinator, because the call site is in a file
# this batch owns. The A-evaluation-4 fix made GroupManager.get_invalid_groups
# disclose that its empty answer is VACUOUS when min_group_size <= 1: no group
# can ever be flagged, because every group is built from a level that actually
# occurs, so no group can hold fewer than one row. That disclosure is right and
# stays in _grouping.py. It is wrong at THIS call site, which asks
# get_invalid_groups for one thing only, the NAMES of the groups that were
# dropped, and reads an empty list as "none were dropped", which at a threshold
# of 1 is true. Measured 2026-09-27: a healthy rerank raised the vacuity warning
# FOUR times, once per get_ranking_group_metrics call.


def test_a_healthy_rerank_over_two_groups_is_not_drowned_in_caveats():
    """The real flow, through the caller that deliberately asks for no filtering.

    ``rerank.py`` calls get_ranking_group_metrics with min_group_size=1 under the
    comment "so every present group is included in this intervention view", four
    times per rerank (before and after, two metrics each). Before: four copies of
    "GroupManager.get_invalid_groups: min_group_size is 1 ... This empty result is
    vacuous". After: nothing at all, because no group was dropped and none could
    have been.
    """
    from vfairness.post_processing.ranking.rerank import exposure_parity_rerank

    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.40, 0.35, 0.30, 0.25, 0.20])
    groups = np.array(["A"] * 5 + ["B"] * 5)
    _, messages = _caught(lambda: exposure_parity_rerank(scores, groups))
    assert messages == [], f"a healthy rerank is warning about something: {messages}"


def test_the_drop_warning_says_nothing_at_a_threshold_that_cannot_drop_anything():
    """The same thing one layer down, without the rerank machinery in the way."""
    from vfairness.evaluation.vfairness_metrics._grouping import GroupManager

    sens = np.array(["A"] * 40 + ["B"])
    for threshold in (0, 1):
        _, messages = _caught(
            lambda t=threshold: C._warn_dropped_groups(GroupManager(sens, min_group_size=t))
        )
        assert messages == [], (threshold, messages)


def test_control_an_undersized_group_is_still_named_at_a_threshold_that_can_flag():
    """OVER-CORRECTION CONTROL: 2 is the smallest threshold that can flag anything.

    A fix that returned early for every threshold would pass the two pins above
    and silence the disclosure this helper exists for, which was itself a defect
    fixed on 2026-09-10 (a 38-level tail holding 47.9 percent of the rows left
    every comparison in silence).
    """
    from vfairness.evaluation.vfairness_metrics._grouping import GroupManager

    sens = np.array(["A"] * 40 + ["B"])
    for threshold in (2, 30):
        _, messages = _caught(
            lambda t=threshold: C._warn_dropped_groups(GroupManager(sens, min_group_size=t))
        )
        assert any(
            f"Excluding 1 group(s) below min_group_size={threshold}: {{'B': 1}}" in m
            for m in messages
        ), (threshold, messages)
        assert any("1 of 41 rows (2.4 percent)" in m for m in messages), (threshold, messages)


def test_worst_group_accuracy_is_silent_at_a_threshold_that_cannot_drop_anything():
    """The SECOND reader of the drop list in classification.py, same reading.

    ``worst_group_accuracy`` asks ``if gm.get_invalid_groups()`` and returns NaN
    when anything was dropped. At a threshold of 1 nothing can be, so the answer
    was always [] and the verdict always went through; the only effect of the call
    was the vacuity warning. Measured before on 20 + 20 rows at accuracy 1.0:
    worst_group_accuracy(min_group_size=1) -> 1.0 with
    "GroupManager.get_invalid_groups: min_group_size is 1 ... vacuous". The CI/CD
    gate reaches this file with min_group_size=1 (gate.py:1288).
    """
    y_true = np.array([1, 0] * 20)
    y_pred = np.array([1, 0] * 20)
    sens = np.array(["A"] * 20 + ["B"] * 20)
    for threshold in (1, 2):
        value, messages = _caught(
            lambda t=threshold: C.worst_group_accuracy(y_true, y_pred, sens, min_group_size=t)
        )
        assert value == pytest.approx(1.0), (threshold, value)
        assert messages == [], (threshold, messages)


def test_control_worst_group_accuracy_still_refuses_a_genuinely_undersized_group():
    """OVER-CORRECTION CONTROL: the size gate itself must still fire and be named.

    35 rows in A and 5 in B at min_group_size=30: the NaN refusal and the drop
    warning are the point of both call sites, and a fix that skipped the check for
    every threshold would delete them.
    """
    sens = np.array(["A"] * 35 + ["B"] * 5)
    value, messages = _caught(
        lambda: C.worst_group_accuracy(
            np.array([1, 0] * 20), np.array([1, 0] * 20), sens, min_group_size=30
        )
    )
    assert math.isnan(value)
    assert any("Excluding 1 group(s) below min_group_size=30: {'B': 5}" in m for m in messages), (
        messages
    )
