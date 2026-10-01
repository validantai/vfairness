"""Surface grading batch g010: vfairness_metrics._grouping.

Every item in this file is pinned against the one defect the batch exists to
find: a value nobody measured, handed back as a neutral default and then read
as a result. Two were live here.

1. ``compute_max_difference`` refused a dict of FEWER THAN TWO GROUPS (that
   guard was already in place) but not a dict of two groups whose values were
   undefined. Every pair was skipped as NaN, the 0.0 initialiser survived, and
   0.0 is perfect equality on a difference scale. Reproduced end to end:
   ``mae_parity_difference`` returned exactly 0.0, its PASS value, on data
   where one group's MAE could not be computed at all.

2. ``GroupManager`` built a group's mask with ``attr == value``, which is False
   on every row when the value is NaN. A missing sensitive attribute therefore
   produced a group of size 0 holding a proportion of 0.0, while the rows were
   really there. The DataFrame path counted the same rows correctly, so the
   two paths answered 0 and 10 for identical data.

Each pin is paired with a healthy-data control whose expected value is computed
here by hand, so a refusal that swallows real measurement fails too.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._grouping import (
    GroupManager,
    compute_max_difference,
)
from vfairness.evaluation.vfairness_metrics.regression import mae_parity_difference
from vfairness.exceptions import ConfigurationError


def _healthy() -> GroupManager:
    """60 rows in group 'a', 40 in group 'b', both over a gate of 30."""
    return GroupManager(np.array(["a"] * 60 + ["b"] * 40), min_group_size=30)


# ---------------------------------------------------------------------------
# compute_max_difference
# ---------------------------------------------------------------------------


def test_max_difference_control_measures_a_real_gap():
    """CONTROL. 0.50 against 0.25 is a gap of 0.25, and 'a' is the higher arm."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diff, high, low = compute_max_difference({"a": 0.50, "b": 0.25})
    assert diff == pytest.approx(0.50 - 0.25)
    assert (high, low) == ("a", "b")
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_max_difference_control_three_groups_picks_the_widest_pair():
    """CONTROL. The widest pair of 0.90, 0.25 and 0.50 is 0.65, high group 'c'."""
    diff, high, low = compute_max_difference({"a": 0.50, "b": 0.25, "c": 0.90})
    assert diff == pytest.approx(0.90 - 0.25)
    assert (high, low) == ("c", "b")


def test_max_difference_control_equal_groups_still_measure_zero():
    """CONTROL. A real tie is a real 0.0 and must NOT be refused as unmeasurable."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diff, high, low = compute_max_difference({"a": 0.50, "b": 0.50})
    assert diff == 0.0
    assert not np.isnan(diff)
    assert (high, low) == ("", "")
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_max_difference_refuses_when_no_pair_is_comparable():
    """Two groups, one undefined value: no pair exists, so NaN and never 0.0."""
    with pytest.warns(UserWarning, match="NOT MEASURABLE"):
        diff, high, low = compute_max_difference({"a": float("nan"), "b": 0.9})
    assert np.isnan(diff)
    assert (high, low) == ("", "")

    with pytest.warns(UserWarning, match="NOT MEASURABLE"):
        diff_all, _, _ = compute_max_difference(
            {"a": float("nan"), "b": float("nan"), "c": float("nan")}
        )
    assert np.isnan(diff_all)


def test_max_difference_refuses_two_infinite_values():
    """inf minus inf is undefined, not a tie: the old code returned 0.0 here."""
    with pytest.warns(UserWarning, match="NOT MEASURABLE"):
        diff, _, _ = compute_max_difference({"a": float("inf"), "b": float("inf")})
    assert np.isnan(diff)


def test_max_difference_keeps_measuring_when_two_values_survive():
    """A partial drop still MEASURES the survivors, and says who fell out.

    This is the half a refusal must not swallow: 0.1 against 0.9 is a real
    0.8 gap and stays one. Only the coverage is disclosed.
    """
    with pytest.warns(UserWarning, match="lower bound"):
        diff, high, low = compute_max_difference({"a": float("nan"), "b": 0.1, "c": 0.9})
    assert diff == pytest.approx(0.9 - 0.1)
    assert (high, low) == ("c", "b")


def test_max_difference_reads_numpy_scalars_as_measurements():
    """np.float32 is a number. Rejecting it would throw real evidence away."""
    diff, high, low = compute_max_difference({"a": np.float32(0.50), "b": np.float32(0.25)})
    assert diff == pytest.approx(0.25, abs=1e-6)
    assert (high, low) == ("a", "b")


def test_mae_parity_returns_nan_not_zero_when_a_group_is_undefined():
    """CALLER-VISIBLE reproduction. Before the fix this returned 0.0.

    Group 'a' has y_true == y_pred == inf, so its per-row error is inf minus
    inf, i.e. undefined, and its MAE is NaN. Group 'b' has a real MAE of 8.0.
    There is nothing to compare, and 0.0 read as "identical error for every
    group", which is the PASS value of this metric.
    """
    n = 40
    attr = np.array(["a"] * n + ["b"] * n)
    y_true = np.concatenate([np.full(n, np.inf), np.full(n, 1.0)])
    y_pred = np.concatenate([np.full(n, np.inf), np.full(n, 9.0)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = mae_parity_difference(y_true, y_pred, attr)
    assert np.isnan(value), f"expected NaN for an unmeasurable MAE gap, got {value!r}"

    # CONTROL on the same shape: a findable disparity is still measured exactly.
    # Group 'a' is predicted perfectly (MAE 0.0), group 'b' is out by 8.0, so
    # the parity difference is 8.0.
    y_true_ok = np.full(2 * n, 1.0)
    y_pred_ok = np.concatenate([np.full(n, 1.0), np.full(n, 9.0)])
    healthy = mae_parity_difference(y_true_ok, y_pred_ok, attr)
    assert healthy == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# GroupManager: sizes, masks, indices, proportions
# ---------------------------------------------------------------------------


def test_group_sizes_masks_and_indices_control():
    """CONTROL. 60 and 40 rows, with the masks and indices that go with them."""
    gm = _healthy()
    assert gm.groups == ["a", "b"]
    assert gm.n_groups == 2
    assert gm.get_group_sizes() == {"a": 60, "b": 40}
    assert gm.get_size("a") == 60
    assert int(gm.get_mask("b").sum()) == 40
    assert gm.get_indices("b").tolist() == list(range(60, 100))
    assert gm.get_group_proportions() == {"a": pytest.approx(0.6), "b": pytest.approx(0.4)}
    assert gm.get_group("a").size == 60
    assert [(name, info.size) for name, info in gm.iter_groups()] == [("a", 60), ("b", 40)]
    assert gm.get_valid_groups() == ["a", "b"]
    assert gm.get_invalid_groups() == []


def test_get_group_raises_for_an_unknown_name():
    """Negative case: an unknown group is an error, never an empty group."""
    with pytest.raises(KeyError):
        _healthy().get_group("does-not-exist")


@pytest.fixture
def missing_gm() -> GroupManager:
    """10 rows at level 1.0 and 10 rows whose sensitive value is missing.

    ``attr == nan`` is False on every row, so this level used to report size 0,
    an empty mask, no indices and a proportion of 0.0 while holding half the
    dataset. One assertion per item below, so each one fails on its own when
    the NaN-aware mask is taken away.
    """
    attr = np.array([1.0] * 10 + [np.nan] * 10)
    with pytest.warns(UserWarning, match="MISSING sensitive attribute"):
        return GroupManager(attr, min_group_size=1)


def test_missing_level_get_size_counts_the_rows(missing_gm):
    assert missing_gm.get_size("nan") == 10


def test_missing_level_group_sizes_count_the_rows(missing_gm):
    assert missing_gm.get_group_sizes() == {"1.0": 10, "nan": 10}


def test_missing_level_mask_selects_the_rows(missing_gm):
    assert int(missing_gm.get_mask("nan").sum()) == 10


def test_missing_level_indices_point_at_the_rows(missing_gm):
    assert missing_gm.get_indices("nan").tolist() == list(range(10, 20))


def test_missing_level_proportion_is_its_real_share(missing_gm):
    assert missing_gm.get_group_proportions()["nan"] == pytest.approx(0.5)


def test_missing_level_get_group_carries_the_real_size(missing_gm):
    assert missing_gm.get_group("nan").size == 10


def test_missing_level_iter_groups_covers_every_row(missing_gm):
    assert sum(info.size for _, info in missing_gm.iter_groups()) == 20


def test_missing_level_get_info_reports_the_real_sizes(missing_gm):
    info = missing_gm.get_info()
    assert info["group_sizes"] == {"1.0": 10, "nan": 10}
    assert sorted(info["valid_groups"]) == ["1.0", "nan"]


def test_missing_level_matches_the_dataframe_path(missing_gm):
    """The DataFrame path always counted these rows correctly (it stringifies
    before comparing). The two paths answering 0 and 10 for identical data is
    what proved 0 wrong."""
    attr = np.array([1.0] * 10 + [np.nan] * 10)
    df_gm = GroupManager(pd.DataFrame({"g": attr}), min_group_size=1)
    assert df_gm.get_group_sizes() == missing_gm.get_group_sizes()


def test_every_row_lands_in_exactly_one_group_when_values_are_missing():
    """The group sizes must add up to the dataset, which is what the coverage
    disclosures in classification.py divide by."""
    attr = np.array([1.0] * 40 + [2.0] * 40 + [np.nan] * 40)
    with pytest.warns(UserWarning, match="MISSING sensitive attribute"):
        gm = GroupManager(attr, min_group_size=30)
    sizes = gm.get_group_sizes()
    assert sum(sizes.values()) == 120
    assert sizes["nan"] == 40
    stacked = np.vstack([gm.get_mask(name) for name in gm.groups])
    assert stacked.sum(axis=0).tolist() == [1] * 120


def test_a_literal_string_nan_is_still_matched_by_value():
    """Negative case: the missing branch must not capture the STRING 'nan'."""
    attr = np.array(["nan"] * 10 + ["a"] * 10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm = GroupManager(attr, min_group_size=1)
    assert gm.get_group_sizes() == {"a": 10, "nan": 10}
    assert [w for w in caught if "MISSING" in str(w.message)] == []


# ---------------------------------------------------------------------------
# GroupManager: the size gate
# ---------------------------------------------------------------------------


def test_all_groups_below_the_gate_returns_empty_and_warns():
    gm = GroupManager(np.array(["a"] * 5 + ["b"] * 4), min_group_size=30)
    with pytest.warns(UserWarning, match="not assessable"):
        assert gm.get_valid_groups() == []
    assert sorted(gm.get_invalid_groups()) == ["a", "b"]
    info = gm.get_info()
    assert info["n_valid_groups"] == 0
    assert info["n_invalid_groups"] == 2
    assert info["group_sizes"] == {"a": 5, "b": 4}


def test_partial_drop_keeps_the_survivors_and_names_the_dropped():
    gm = GroupManager(np.array(["a"] * 60 + ["b"] * 4), min_group_size=30)
    assert gm.get_valid_groups() == ["a"]
    assert gm.get_invalid_groups() == ["b"]
    # The dropped group keeps its real size, so a caller can disclose the rows.
    assert gm.get_group_sizes()["b"] == 4


# ---------------------------------------------------------------------------
# GroupManager: statistics and rates
# ---------------------------------------------------------------------------


def test_compute_group_statistic_control_and_undefined_values():
    gm = _healthy()
    values = np.array([1.0] * 60 + [5.0] * 40)
    assert gm.compute_group_statistic(values, "mean") == {"a": 1.0, "b": 5.0}

    # A group whose values are all undefined gets NaN, not 0.0.
    broken = values.copy()
    broken[:60] = np.nan
    stats = gm.compute_group_statistic(broken, "mean")
    assert np.isnan(stats["a"])
    assert stats["b"] == 5.0

    with pytest.raises(ConfigurationError):
        gm.compute_group_statistic(values, "not-a-statistic")


def test_compute_group_rate_control_and_empty_denominator():
    gm = _healthy()
    # 30 of the 60 'a' rows and 10 of the 40 'b' rows are positive.
    numerator = np.array([True] * 30 + [False] * 30 + [True] * 10 + [False] * 30)
    rates = gm.compute_group_rate(numerator)
    assert rates["a"] == pytest.approx(30 / 60)
    assert rates["b"] == pytest.approx(10 / 40)

    # Conditioning on a subset that excludes group 'b' entirely leaves it with
    # an empty denominator: NaN, never 0.0 ("nobody in that group selected").
    denominator = np.array([True] * 60 + [False] * 40)
    conditioned = gm.compute_group_rate(numerator, denominator_mask=denominator)
    assert conditioned["a"] == pytest.approx(30 / 60)
    assert np.isnan(conditioned["b"])


def test_group_rate_of_zero_is_still_a_measurement():
    """CONTROL for the refusal above: a real 0.0 rate must survive it."""
    gm = _healthy()
    numerator = np.array([True] * 60 + [False] * 40)
    rates = gm.compute_group_rate(numerator)
    assert rates["a"] == 1.0
    assert rates["b"] == 0.0
    assert not np.isnan(rates["b"])
    diff, high, low = compute_max_difference(rates)
    assert diff == pytest.approx(1.0)
    assert (high, low) == ("a", "b")


def test_get_info_agrees_with_the_parts_it_summarises():
    gm = _healthy()
    info = gm.get_info()
    assert info == {
        "n_groups": 2,
        "n_valid_groups": 2,
        "n_invalid_groups": 0,
        "is_intersectional": False,
        "min_group_size": 30,
        "group_sizes": {"a": 60, "b": 40},
        "valid_groups": ["a", "b"],
        "invalid_groups": [],
    }
