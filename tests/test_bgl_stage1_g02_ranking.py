"""Beta Go-Live stage 1, group g02: the ranking metrics fabricated THE RANKING.

The defect, reproduced at the public API before the fix (all values real):

    exposure_parity_ratio(np.full(12, np.nan), ['A']*6 + ['B']*6)
        -> 0.5410755479122771      # BIT-FOR-BIT the genuinely unfair ranking
    exposure_parity_ratio(real_scores, same groups)
        -> 0.5410755479122771      # identical
    exposure_parity_ratio(np.full(12, np.nan), ['A','B']*6)
        -> 0.8187325395606262      # same non-data, four-fifths PASS not FAIL

    normalized_discounted_kl_divergence(np.full(20, np.nan), ['A']*10+['B']*10)
        -> 0.48508869906636626     # == the real np.arange(20) ranking
    attention_weighted_rank_fairness(np.full(40, np.nan), ['A']*20+['B']*20)
        -> value=1.3635184916013479, is_fair=False

Zero warnings on every one of those runs.

Root cause: ``_rankings_to_positions`` fell through to
``np.argsort(-arr, kind="stable")`` for anything that is not an exact
permutation. A stable descending argsort of an ALL-TIED or ALL-NaN column
returns the IDENTITY, i.e. the caller's own ROW ORDER, so the metrics measured
the fairness of an ordering nobody produced -- proved by the same data swinging
0.485 to 0.105 on a permutation of the rows alone.

The fix is three states, never two: the shared helper refuses ABOVE the
dispatch, and each of the five callers turns that refusal into NaN (and
``is_fair=None`` where it carries a verdict) with a ``UserWarning`` naming the
reason. Every test below asserts at the PUBLIC entry point (``vfairness.<fn>``),
and every one is paired with a CONTROL proving healthy data still measures.
"""

import math
import warnings

import numpy as np
import pytest

import vfairness as vf
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)

# ---------------------------------------------------------------------------
# The fabricated values, recorded so a regression cannot quietly return one.
# ---------------------------------------------------------------------------
FABRICATED_RATIO = 0.5410755479122771
FABRICATED_NDKL = 0.48508869906636626
FABRICATED_ATTENTION = 1.3635184916013479

GROUPED_12 = np.array(["A"] * 6 + ["B"] * 6)
INTERLEAVED_12 = np.array(["A", "B"] * 6)
REAL_SCORES_12 = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.4, 0.35, 0.3, 0.2, 0.1, 0.05])

GROUPED_20 = np.array(["A"] * 10 + ["B"] * 10)
INTERLEAVED_20 = np.array(["A", "B"] * 10)
REAL_POSITIONS_20 = np.arange(20)

GROUPED_40 = np.array(["A"] * 20 + ["B"] * 20)
REAL_POSITIONS_40 = np.arange(40, dtype=float)[::-1]

DEGENERATE_12 = {
    "all_nan": np.full(12, np.nan),
    "all_tied": np.full(12, 0.5),
    # A different constant, because 0.5 is a power of two: np.var of a constant
    # array is exactly 0.0 only at some n and some values, which is how a
    # variance-based guard passes for the fixture and fails on real data. The
    # guard uses np.unique on the RAW column, so the constant does not matter.
    "all_tied_0_3": np.full(12, 0.3),
    "all_inf": np.full(12, np.inf),
}


def _quiet(fn, *args, **kwargs):
    """Call fn with warnings silenced; the VALUE is the subject here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# exposure_parity_ratio -- the four-fifths screen, the one that reaches a graded
# and rendered surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(DEGENERATE_12))
def test_exposure_parity_ratio_refuses_a_score_column_with_no_order(case):
    """A column that defines no order is could-not-check, not a ratio."""
    scores = DEGENERATE_12[case]

    with pytest.warns(UserWarning, match="no ranking order is defined"):
        ratio = vf.exposure_parity_ratio(scores, GROUPED_12)

    assert math.isnan(ratio), f"{case}: expected NaN, got {ratio!r}"
    # The precise fabrication: the number was bit-for-bit the real unfair one.
    assert ratio != pytest.approx(FABRICATED_RATIO)
    # And it must not have become the perfect-parity sentinel either.
    assert ratio != 1.0


@pytest.mark.parametrize("case", sorted(DEGENERATE_12))
def test_exposure_parity_ratio_no_longer_depends_on_caller_row_order(case):
    """The same non-data gave 0.541 grouped and 0.819 interleaved: FAIL vs PASS."""
    scores = DEGENERATE_12[case]

    grouped = _quiet(vf.exposure_parity_ratio, scores, GROUPED_12)
    interleaved = _quiet(vf.exposure_parity_ratio, scores, INTERLEAVED_12)

    assert math.isnan(grouped) and math.isnan(interleaved)


def test_exposure_parity_ratio_refuses_a_single_missing_score():
    """argsort sorts NaN LAST, so one missing score is silently ranked worst."""
    scores = REAL_SCORES_12.copy()
    scores[0] = np.nan

    with pytest.warns(UserWarning, match="1 of 12 ranking scores are not finite"):
        ratio = vf.exposure_parity_ratio(scores, GROUPED_12)

    assert math.isnan(ratio)


def test_exposure_parity_ratio_control_healthy_scores_still_measure():
    """CONTROL: a fix that makes everything refuse is a worse defect."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a healthy column must warn about NOTHING
        ratio = vf.exposure_parity_ratio(REAL_SCORES_12, GROUPED_12)

    assert ratio == pytest.approx(0.5410755479122771, rel=1e-12)
    assert not math.isnan(ratio)


def test_exposure_parity_ratio_control_positions_still_measure():
    """CONTROL: the integer-permutation path is untouched by the guard."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio = vf.exposure_parity_ratio(np.arange(12), GROUPED_12)

    assert ratio == pytest.approx(0.5410755479122771, rel=1e-12)


def test_exposure_parity_ratio_reaches_the_grader_as_could_not_check():
    """The consumer surface: NaN routes to COULD_NOT_CHECK, 0.541 still FAILS.

    Rule 1 of the fix rules: removing the fabricated number must not remove the
    alarm. The healthy reading is still a four-fifths FAIL.
    """
    unmeasurable = _quiet(vf.exposure_parity_ratio, DEGENERATE_12["all_nan"], GROUPED_12)
    measured = _quiet(vf.exposure_parity_ratio, REAL_SCORES_12, GROUPED_12)

    assert check_threshold("exposure_parity_ratio", float(unmeasurable), 0.80)[0] is (
        ThresholdOutcome.COULD_NOT_CHECK
    )
    assert check_threshold("exposure_parity_ratio", float(measured), 0.80)[0] is (
        ThresholdOutcome.FAIL
    )


# ---------------------------------------------------------------------------
# normalized_discounted_kl_divergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores",
    [np.full(20, np.nan), np.full(20, 0.5), np.full(20, 0.3), np.full(20, -np.inf)],
    ids=["all_nan", "all_tied", "all_tied_0_3", "all_neg_inf"],
)
def test_ndkl_refuses_a_score_column_with_no_order(scores):
    """NDKL is POSITION-DISCOUNTED: with no order, the discounts are row order."""
    with pytest.warns(UserWarning, match="no ranking order is defined"):
        ndkl = vf.normalized_discounted_kl_divergence(scores, GROUPED_20)

    assert math.isnan(ndkl)
    assert ndkl != pytest.approx(FABRICATED_NDKL)
    assert ndkl != 0.0


def test_ndkl_no_longer_swings_on_a_permutation_of_the_rows():
    """0.48508869906636626 grouped vs 0.10479033130746856 interleaved, same data."""
    scores = np.full(20, np.nan)

    grouped = _quiet(vf.normalized_discounted_kl_divergence, scores, GROUPED_20)
    interleaved = _quiet(vf.normalized_discounted_kl_divergence, scores, INTERLEAVED_20)

    assert math.isnan(grouped) and math.isnan(interleaved)


def test_ndkl_control_healthy_ranking_still_measures():
    """CONTROL: the real shut-out ranking still reports its real divergence."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ndkl = vf.normalized_discounted_kl_divergence(REAL_POSITIONS_20, GROUPED_20)

    assert ndkl == pytest.approx(0.48508869906636626, rel=1e-12)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fair = vf.normalized_discounted_kl_divergence(REAL_POSITIONS_20, INTERLEAVED_20)

    # CONTROL, second half: the metric still DISCRIMINATES between a shut-out
    # ranking and a fair one. A blanket NaN would pass the assertion above only
    # by accident; this pair cannot both be satisfied by a refusal.
    assert fair == pytest.approx(0.10479033130746856, rel=1e-12)
    assert fair < ndkl


# ---------------------------------------------------------------------------
# attention_weighted_rank_fairness -- the one that attaches a VERDICT.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores",
    [np.full(40, np.nan), np.full(40, 0.5), np.full(40, 0.3)],
    ids=["all_nan", "all_tied", "all_tied_0_3"],
)
def test_attention_weighted_refuses_and_withholds_the_verdict(scores):
    """value NaN AND is_fair None: could-not-check, never a confident FAIL."""
    with pytest.warns(UserWarning, match="no ranking order is defined"):
        result = vf.attention_weighted_rank_fairness(scores, GROUPED_40)

    assert math.isnan(result.value)
    assert result.value != pytest.approx(FABRICATED_ATTENTION)
    # Three states, never two: None is could-not-check. False would be a graded
    # FAIL and True a graded PASS, and adapters_ranking reads this field first.
    assert result.is_fair is None
    assert result.is_fair is not False

    # The group inventory IS measured, so the groups are still named; every
    # POSITION-derived number beside them is withheld.
    assert set(result.group_exposures) == {"A", "B"}
    assert all(math.isnan(v) for v in result.group_exposures.values())
    assert all(math.isnan(v) for v in result.group_attentions.values())


def test_attention_weighted_control_healthy_ranking_still_grades():
    """CONTROL: a real ranking still gets its real number and a real verdict."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = vf.attention_weighted_rank_fairness(REAL_POSITIONS_40, GROUPED_40)

    assert result.value == pytest.approx(1.363518491601348, rel=1e-12)
    assert result.is_fair is False
    assert result.group_exposures["A"] == pytest.approx(0.3182407541993259, rel=1e-12)
    assert result.group_exposures["B"] == pytest.approx(1.681759245800674, rel=1e-12)


def test_attention_weighted_control_still_separates_shut_out_from_interleaved():
    """CONTROL: the guard did not turn every answer into a refusal.

    Both of these are GRADED (a real bool, not the could-not-check None) and
    the shut-out ranking still scores far worse than the interleaved one, so a
    blanket NaN cannot satisfy this test. ``is_fair`` is False for both because
    this metric's own threshold is 0.1 and its position-attention model is
    top-heavy; the subject here is that a verdict was REACHED, not which one.
    """
    interleaved_groups = np.array(["A", "B"] * 20)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shut_out = vf.attention_weighted_rank_fairness(REAL_POSITIONS_40, GROUPED_40)
        interleaved = vf.attention_weighted_rank_fairness(np.arange(40), interleaved_groups)

    assert isinstance(shut_out.is_fair, bool) and isinstance(interleaved.is_fair, bool)
    assert not math.isnan(shut_out.value) and not math.isnan(interleaved.value)
    assert interleaved.value == pytest.approx(0.31824075419932607, rel=1e-12)
    assert interleaved.value < shut_out.value


# ---------------------------------------------------------------------------
# The shared root: all five callers of _rankings_to_positions, guarded ABOVE the
# dispatch so fixing one does not move the fabrication to its siblings.
# ---------------------------------------------------------------------------


def test_every_ranking_metric_refuses_the_same_unreadable_column():
    """Rule 3: one precondition, guarded once, inherited by every branch."""
    scores = np.full(12, np.nan)

    assert math.isnan(_quiet(vf.exposure_parity_difference, scores, GROUPED_12))
    assert math.isnan(_quiet(vf.exposure_parity_ratio, scores, GROUPED_12))
    assert math.isnan(_quiet(vf.normalized_discounted_kl_divergence, scores, GROUPED_12))
    assert math.isnan(_quiet(vf.attention_weighted_rank_fairness, scores, GROUPED_12).value)

    metrics = _quiet(vf.get_ranking_group_metrics, scores, GROUPED_12)
    # The count is real and survives; every POSITION-derived cell is withheld.
    assert metrics["A"]["count"] == 6
    assert metrics["B"]["count"] == 6
    for group in ("A", "B"):
        for cell in (
            "avg_position",
            "avg_exposure",
            "min_position",
            "max_position",
            "median_position",
        ):
            assert math.isnan(metrics[group][cell]), f"{group}.{cell} was not withheld"


def test_get_ranking_group_metrics_control_still_measures():
    """CONTROL for the fifth caller."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = vf.get_ranking_group_metrics(np.arange(12), GROUPED_12)

    assert metrics["A"]["count"] == 6
    assert metrics["A"]["avg_position"] == pytest.approx(2.5)
    assert metrics["B"]["avg_position"] == pytest.approx(8.5)
    assert metrics["A"]["avg_exposure"] > metrics["B"]["avg_exposure"]
