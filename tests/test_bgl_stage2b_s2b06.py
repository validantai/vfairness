"""Beta Go-Live stage 2b, group s2b06: data_balancing.py and Resampler.

Stage 2 closed nine fabricated measurements in
``preprocessing/feature_engineering/data_balancing.py``. An independent audit of
those fixes found that the same defect was still live in a SIBLING of almost
every one of them, and that two of the fixes were not pinned by anything. This
module is that second pass. Every test below names the value that was fabricated
and the value that is returned now, both measured at the PUBLIC entry point, and
every group of tests carries a control proving that healthy data still produces
the same measurement it always did.

The defect class, in one sentence: a value nobody measured, replaced by a neutral
default, then graded, counted, compared or reported as if it were a measurement.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.data_balancing import (
    ADASYNResampler,
    CounterfactualAugmenter,
    InversePropensityWeighter,
    PropensityScoreWeighter,
    SMOTEResampler,
    TomekResampler,
    propensity_weights,
)
from vfairness.preprocessing.feature_engineering.transformers import Resampler

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _two_group_frame(n_a: int = 80, n_b: int = 20, seed: int = 11) -> pd.DataFrame:
    """80/20 groups with a feature that genuinely separates them, so the
    logistic regression has something to fit and the weights are not uniform."""
    rng = np.random.default_rng(seed)
    grp = np.array(["A"] * n_a + ["B"] * n_b)
    return pd.DataFrame(
        {
            "grp": grp,
            "x1": np.where(grp == "A", 0.0, 1.0) + rng.normal(0, 0.5, n_a + n_b),
            "x2": rng.normal(size=n_a + n_b),
        }
    )


def _single_group_frame(n: int = 40, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"grp": ["A"] * n, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})


def _constant_feature_frame(n: int = 40) -> pd.DataFrame:
    """Two groups, but no feature varies: nothing can be estimated from it."""
    return pd.DataFrame(
        {"grp": ["A"] * (n // 2) + ["B"] * (n // 2), "x1": [1.0] * n, "x2": [2.0] * n}
    )


def _solver_failure_frame() -> pd.DataFrame:
    """One np.inf cell; the logistic-regression solver refuses the matrix."""
    broken = _two_group_frame()
    broken.loc[0, "x1"] = np.inf
    return broken


def _nine_identical_plus_one(seed: int = 7):
    """The shape the old `n_distinct < 2` predicate could not see: the degenerate
    cell holds 9 byte-identical rows AND one different row, so it has TWO
    distinct rows and passed the input-side check, while every interpolation that
    picked two of the nine returned one of them unchanged."""
    rng = np.random.default_rng(seed)
    n_a = 40
    df = pd.DataFrame(
        {
            "f0": list(rng.normal(0, 1, n_a)) + [5.0] * 9 + [9.0],
            "f1": list(rng.normal(0, 1, n_a)) + [7.0] * 9 + [1.0],
            "group": ["a"] * n_a + ["b"] * 10,
        }
    )
    y = np.array([i % 2 for i in range(n_a)] + [1] * 10)
    return df, y


def _five_identical_rows(seed: int = 3):
    """The degenerate cell holds FIVE real rows that are numerically identical,
    so a copy count taken from `len(cell)` reads 5 and reports nothing."""
    rng = np.random.default_rng(seed)
    n_a = 30
    df = pd.DataFrame(
        {
            "f0": list(rng.normal(0, 1, n_a)) + [4.0] * 5,
            "f1": list(rng.normal(0, 1, n_a)) + [8.0] * 5,
            "group": ["a"] * n_a + ["b"] * 5,
        }
    )
    y = np.array([i % 2 for i in range(n_a)] + [1] * 5)
    return df, y


def _one_row_cell_frame(seed: int = 4):
    """59 rows in group a, ONE in group b: the (b, label=1) cell holds a single
    real observation, so 29 of the 30 rows it comes back with are copies."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "f0": np.concatenate([rng.normal(0, 1, 59), [5.0]]),
            "f1": np.concatenate([rng.normal(0, 1, 59), [7.0]]),
            "group": ["a"] * 59 + ["b"],
        }
    )
    y = np.array([i % 2 for i in range(59)] + [1])
    return df, y


def _healthy_cells_frame(n_a: int = 250, n_b: int = 50, seed: int = 0):
    """Every (group, label) cell is populated and none is degenerate."""
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "grp": np.array(["A"] * n_a + ["B"] * n_b),
        }
    )
    y = np.concatenate([np.tile([0, 1], n_a // 2), np.tile([0, 1], n_b // 2)])
    return X, y


def _no_numeric_feature_frame():
    X = pd.DataFrame({"c1": ["p", "q"] * 20, "group": ["a"] * 20 + ["b"] * 20})
    return X, np.array([0, 1] * 20)


def _joined(messages) -> str:
    return " ".join(messages)


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught]


# ===========================================================================
# 1. get_sample_weights: the branch nothing reached, and the single-group
#    early return that handed 1.0 to groups the model never saw.
# ===========================================================================


@pytest.mark.parametrize(
    "frame, expected_status",
    [(_constant_feature_frame(), "not_estimable"), (_solver_failure_frame(), "solver_failed")],
    ids=["not_estimable", "solver_failed"],
)
def test_get_sample_weights_refuses_a_model_that_was_never_estimated(frame, expected_status):
    """The unpinned branch. Before this test, `return np.ones(len(df))` could
    replace the NaN branch and the whole suite stayed green while the public
    entry handed back neutral weights for a model that was never estimated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = InversePropensityWeighter(protected_attributes=["grp"]).fit(frame)

    # The fixture must actually reach the branch this test is about.
    assert t.fit_result.fit_metrics["propensity_model"] == expected_status, (
        "fixture no longer produces an unestimated model"
    )

    with pytest.warns(UserWarning, match="no propensity model was estimated"):
        weights = t.get_sample_weights(frame)

    assert len(weights) == len(frame)
    assert np.isnan(weights).all(), (
        f"a model with status {expected_status!r} handed back weights {weights[:5]!r}; "
        "a weight that was never derived must not train a model as though it had been"
    )


def test_a_single_group_model_refuses_the_groups_it_never_saw():
    """SIBLING. The single_group early return did `return np.ones(len(df))` for
    the WHOLE batch and said "there is no second group to rebalance against"
    about batches that had several. Measured before: [1.0, 1.0, 1.0, 1.0] for a
    batch of A, B, B, C fitted on A alone."""
    fit_frame = _single_group_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = InversePropensityWeighter(protected_attributes=["grp"]).fit(fit_frame)
    assert t.fit_result.fit_metrics["propensity_model"] == "single_group", (
        "fixture no longer reaches the single-group branch"
    )

    batch = pd.DataFrame(
        {"grp": ["A", "B", "B", "C"], "x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 1.1, 1.2, 1.3]}
    )
    weights, messages = _caught(lambda: t.get_sample_weights(batch))

    assert weights[0] == 1.0, "the row of the ONE observed group lost its identity weight"
    assert np.isnan(weights[1:]).all(), (
        f"rows of groups the model never saw came back as {weights[1:]!r}, not NaN"
    )
    warned = _joined(messages)
    assert "never observed at fit time" in warned
    assert "['B', 'C']" in warned, f"the refusal does not name the unseen groups: {warned!r}"
    assert "there is no second group" not in warned, (
        "the warning still claims there is no second group about a batch that has three"
    )


def test_control_a_single_group_batch_keeps_the_identity_weight():
    """Over-correction control. With one group the identity 1.0 IS the complete
    and correct answer; a fix that NaNs it would break the one case that works."""
    frame = _single_group_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = InversePropensityWeighter(protected_attributes=["grp"]).fit(frame)

    with pytest.warns(UserWarning, match="no propensity model was estimated"):
        weights = t.get_sample_weights(frame)
    assert np.allclose(weights, 1.0), f"the identity case started refusing: {weights[:5]!r}"
    assert t.fit_result.fit_metrics["group_balance"]["max_dev_after"] is None


def test_a_batch_without_the_protected_column_raises_the_named_error():
    """It died as `KeyError: 'grp'` three frames deep inside the group key."""
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)
    with pytest.raises(ValueError, match=r"Protected attributes not found: \['grp'\]"):
        t.get_sample_weights(df.drop(columns=["grp"]).head(5))


# ===========================================================================
# 2. The apply path imputed an unmeasured feature cell to 0.0 and returned a
#    confident weight for it.
# ===========================================================================


def test_an_unmeasured_feature_cell_does_not_earn_a_confident_weight():
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)

    measured_row = df.iloc[[80]]
    gap_row = measured_row.copy()
    gap_row.loc[:, "x1"] = np.nan

    measured_weight = t.get_sample_weights(measured_row)
    # Control first: the row that WAS measured keeps the number it always had.
    assert measured_weight[0] == pytest.approx(4.41485972, abs=1e-6), (
        f"the measured row's weight moved to {measured_weight[0]!r}"
    )

    weights, messages = _caught(lambda: t.get_sample_weights(gap_row))
    # Before: 23.944332, finite, positive, and silent. The imputed 0.0 put the
    # row deep into the other group's region of the feature space.
    assert np.isnan(weights[0]), (
        f"a row whose x1 was never measured earned the weight {weights[0]!r}"
    )
    warned = _joined(messages)
    assert "no measured value in a column the model was fitted on" in warned
    assert "'x1'" in warned


def test_a_row_with_no_recorded_group_is_not_a_group_named_nan():
    """`_group_key_values` stringifies, so an unrecorded protected value became a
    group literally named 'nan'. Measured before: 3 groups, fitted propensities
    and weights of 8.37 to 28.02 for the five unrecorded rows, and
    max_dev_after = 0.033784 reported with `warnings == []`."""
    rng = np.random.default_rng(5)
    n = 60
    grp = np.array(["A"] * 30 + ["B"] * 25 + [np.nan] * 5, dtype=object)
    X = pd.DataFrame({"grp": grp, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    X.loc[X["grp"] == "B", "x1"] += 1.5
    assert X["grp"].isna().sum() == 5, "fixture no longer has unrecorded rows"

    result, messages = _caught(lambda: propensity_weights(X, ["grp"]))

    assert result["group_balance"]["groups"] == ["A", "B"], (
        f"an unrecorded value became a group: {result['group_balance']['groups']}"
    )
    assert result["group_balance"]["n_groups"] == 2
    weights = np.asarray(result["sample_weights"], dtype=float)
    assert np.isnan(weights[55:]).all(), (
        f"rows with no recorded group were weighed {weights[55:]!r}"
    )
    assert np.isfinite(weights[:55]).all(), "the rows that DO have a group lost their weights"
    assert result["n_rows_without_a_recorded_group"] == 5
    assert result["group_balance"]["n_rows_scored"] == 55
    warned = _joined(result["warnings"])
    assert "no recorded value for the protected attribute" in warned
    assert any("no recorded value" in m for m in messages)


def test_an_empty_frame_reports_a_state_instead_of_crashing():
    """n_groups == 0 escaped the vocabulary entirely and died as
    `IndexError: list index out of range` from `groups[0]` in an f-string."""
    empty = pd.DataFrame({"grp": pd.Series([], dtype=object), "x1": pd.Series([], dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = propensity_weights(empty, ["grp"])

    assert result["propensity_model"] == "no_groups"
    assert (
        result["propensity_model_reason"]
        and "no row carries a recorded value" in (result["propensity_model_reason"])
    )
    assert result["sample_weights"] == []
    assert result["group_balance"]["max_dev_after"] is None
    assert result["group_balance"]["share_after"] is None


def test_a_group_with_no_measurable_row_refuses_instead_of_scoring_the_rest():
    """The `share_after[g] = ... if total_w > 0 else uniform` line handed out a
    PERFECT score as its fallback. The replacement refuses: a weighted share is
    not measurable when a whole group has no measured weight."""
    df = _two_group_frame()
    df.loc[df["grp"] == "B", "x1"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = propensity_weights(df, ["grp"])

    assert result["propensity_model"] == "fitted", "fixture no longer fits a model at all"
    assert result["group_balance"]["max_dev_after"] is None, (
        "a balance was reported for a comparison one side of which was never measured"
    )
    assert result["group_balance"]["share_after"] is None
    reason = result["group_balance"]["max_dev_after_reason"]
    assert reason and "no row with a measured weight" in reason
    # max_dev_BEFORE is a fact about the rows that were read and must survive.
    assert result["group_balance"]["max_dev_before"] == pytest.approx(0.30, abs=1e-9)


def test_control_the_healthy_propensity_run_is_the_same_number_it_was():
    """Over-correction control for every refusal above."""
    df = _two_group_frame()
    t = PropensityScoreWeighter(protected_attributes=["grp"]).fit(df)
    balance = t.fit_result.fit_metrics["group_balance"]

    assert t.fit_result.fit_metrics["propensity_model"] == "fitted"
    assert balance["max_dev_before"] == pytest.approx(0.30, abs=1e-9)
    assert balance["max_dev_after"] == pytest.approx(0.119828, abs=1e-6)
    assert balance["share_after"]["A"] == pytest.approx(0.6198278, abs=1e-6)
    assert balance["n_rows_scored"] == 100
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"

    weights = t.get_sample_weights(df)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights.mean() == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 3. SMOTE / ADASYN: "balanced by duplication" was measured on the INPUT.
# ===========================================================================


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_a_cell_of_nine_identical_rows_plus_one_says_the_rows_are_copies(cls):
    """PIN HOLE A, the shape `n_distinct < 2` could not see. The cell has TWO
    distinct rows, so the old predicate said nothing, and 10 of 10 added rows
    came back as verbatim copies under the sentence "Every cell that does exist
    was balanced to 20 rows"."""
    df, y = _nine_identical_plus_one()
    assert len(df[df["group"] == "b"][["f0", "f1"]].drop_duplicates()) == 2, (
        "fixture must hold TWO distinct rows, or it pins the old predicate instead"
    )

    t = cls(protected_attributes=["group"], random_state=42).fit(df, y)
    (X_res, _), messages = _caught(lambda: t.get_resampled_data(df, y))

    b_rows = X_res[X_res["group"] == "b"]
    assert len(b_rows) == 20, "fixture no longer reaches the oversampling path"
    n_copies = len(b_rows) - len(b_rows[["f0", "f1"]].drop_duplicates())
    assert n_copies >= 10, f"fixture no longer produces duplicated output rows (only {n_copies})"

    warned = _joined(t.fit_result.warnings)
    assert "DUPLICATION" in warned, (
        f"10 verbatim copies were returned as resampled data, unreported: {warned!r}"
    )
    # PIN HOLE B: the DISCLOSED NUMBERS, not merely the presence of a string.
    assert "(b, label=1): 2 distinct real row(s) in 10, 10 of 10 added row(s)" in warned, (
        f"the disclosure does not carry the measured counts: {warned!r}"
    )
    assert "effective sample size is unchanged" in warned
    assert any("DUPLICATION" in m for m in messages)
    assert "Every cell that does exist was balanced to 20 rows." not in warned, (
        "the misleading closing sentence is still shipped next to the duplication"
    )


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_a_cell_of_five_identical_real_rows_says_the_rows_are_copies(cls):
    """PIN HOLE A, second shape: a copy count taken from `len(cell)` reads 5 and
    reports nothing, while every added row is a copy."""
    df, y = _five_identical_rows()
    b = df[df["group"] == "b"]
    assert len(b) == 5 and len(b[["f0", "f1"]].drop_duplicates()) == 1, (
        "fixture must hold FIVE numerically identical real rows"
    )

    t = cls(protected_attributes=["group"], random_state=42).fit(df, y)
    X_res, _ = t.get_resampled_data(df, y)

    b_rows = X_res[X_res["group"] == "b"]
    assert len(b_rows) == 15 and len(b_rows[["f0", "f1"]].drop_duplicates()) == 1

    warned = _joined(t.fit_result.warnings)
    assert "(b, label=1): 1 distinct real row(s) in 5, 10 of 10 added row(s)" in warned, (
        f"the disclosure does not carry the measured counts: {warned!r}"
    )


def test_the_one_row_cell_disclosure_carries_its_real_numbers():
    """PIN HOLE B on the shape stage 2 already fixed: the test that pinned it
    asserted only string presence, so hard-coding the counts shipped green."""
    df, y = _one_row_cell_frame()
    t = SMOTEResampler(protected_attributes=["group"], random_state=42).fit(df, y)
    X_res, _ = t.get_resampled_data(df, y)

    b_rows = X_res[X_res["group"] == "b"]
    assert len(b_rows) == 30 and len(b_rows[["f0", "f1"]].drop_duplicates()) == 1
    warned = _joined(t.fit_result.warnings)
    assert "(b, label=1): 1 distinct real row(s) in 1, 29 of 29 added row(s)" in warned, warned
    assert "29 row(s) in total" in warned


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler])
def test_control_a_healthy_resample_still_synthesises_and_says_nothing(cls):
    """A warning on every run is a warning on none."""
    X, y = _healthy_cells_frame()
    t = cls(protected_attributes=["grp"], random_state=42).fit(X, y)
    (X_res, y_res), messages = _caught(lambda: t.get_resampled_data(X, y))

    counts = Counter(X_res["grp"])
    assert len(set(counts.values())) == 1, f"groups came back uneven: {dict(counts)}"
    assert counts["B"] > Counter(X["grp"])["B"], "the minority group did not grow"
    assert len(X_res) == len(y_res)
    b_rows = X_res[X_res["grp"] == "B"]
    assert len(b_rows[["f1", "f2"]].drop_duplicates()) > 200, "synthesis produced copies"
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"
    assert messages == []


# ===========================================================================
# 4. Tomek: the guard sat BELOW the dispatch, and _tomek had no channel at all.
# ===========================================================================


@pytest.mark.parametrize("cls", [SMOTEResampler, ADASYNResampler, TomekResampler])
def test_every_method_discloses_a_frame_with_no_numeric_feature_column(cls):
    """The no-numeric guard used to sit below the tomek dispatch, so Tomek
    returned 40 rows in / 40 rows out with `fit_result.warnings == []` while
    SMOTE on the identical frame disclosed."""
    X, y = _no_numeric_feature_frame()
    t = cls(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, y_res), messages = _caught(lambda: t.get_resampled_data(X, y))

    assert len(X_res) == len(X), "fixture no longer reaches the no-op branch"
    warned = _joined(t.fit_result.warnings)
    assert "none of the feature columns is numeric" in warned, (
        f"{cls.__name__} returned the frame unchanged and reported nothing: {warned!r}"
    )
    assert "returned unchanged" in warned


def test_tomek_does_not_read_an_unmeasured_column_as_an_observation_of_zero():
    """`.fillna(0.0)` decided which rows to DELETE from an invented observation.
    Group b's f0 was never measured and sat at exactly 0.0, far from the
    measured cloud around 50, which made b's rows each other's nearest
    neighbours instead of the a rows' neighbours."""
    rng = np.random.default_rng(2)
    n_a, n_b = 30, 10
    X = pd.DataFrame(
        {
            "f0": np.concatenate([rng.normal(50, 1, n_a), np.full(n_b, np.nan)]),
            "group": ["a"] * n_a + ["b"] * n_b,
        }
    )
    y = np.array([0, 1] * ((n_a + n_b) // 2))
    assert X.loc[X["group"] == "b", "f0"].isna().all(), "fixture: b's f0 must be unmeasured"

    t = TomekResampler(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, _), messages = _caught(lambda: t.get_resampled_data(X, y))

    warned = _joined(t.fit_result.warnings)
    assert "never measured" in warned and "'f0'" in warned, (
        f"an unmeasured column decided deletions in silence: {warned!r}"
    )
    assert "not an observation of 0.0" in warned
    assert f"in {n_b} of {n_a + n_b} row(s)" in warned
    assert any("never measured" in m for m in messages)
    # And the substituted values never reach the caller: f0 is still unmeasured
    # for every surviving b row, not a zero and not a mean.
    assert X_res.loc[X_res["group"] == "b", "f0"].isna().all(), (
        "a substituted neighbour-search value was written into the returned frame"
    )


def test_tomek_refuses_when_nothing_was_measured_anywhere():
    X = pd.DataFrame({"f0": [np.nan] * 10, "group": ["a"] * 5 + ["b"] * 5})
    y = np.array([0, 1] * 5)
    t = TomekResampler(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, _), _ = _caught(lambda: t.get_resampled_data(X, y))

    assert len(X_res) == 10, "rows were deleted on a distance nobody could compute"
    warned = _joined(t.fit_result.warnings)
    assert "not one value was measured" in warned
    assert "returned unchanged" in warned


def test_tomek_never_cleans_a_protected_group_away():
    """The class-preservation rescue existed; the GROUP one, which is the
    fairness-relevant half, did not. A group cleaned away is absent from every
    fairness number computed on the result."""
    f = [0.0, 0.01, 1.0, 1.01, 2.0, 2.01, 5.0, 5.5, 6.0, 6.5]
    X = pd.DataFrame(
        {"f0": f, "f1": f, "group": ["b", "a", "b", "a", "b", "a", "a", "a", "a", "a"]}
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    assert Counter(X["group"]) == Counter({"a": 7, "b": 3})

    t = TomekResampler(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, y_res), messages = _caught(lambda: t.get_resampled_data(X, y))

    after = Counter(X_res["group"])
    assert after["b"] == 3, f"group b was cleaned away: {dict(after)}"
    assert len(X_res) == len(y_res)
    warned = _joined(t.fit_result.warnings)
    assert "would have removed EVERY row of protected group(s) ['b']" in warned, warned
    assert "no row was removed" in warned, "the resulting no-op was not disclosed"
    assert "no Tomek link was found" not in warned, (
        "it claims no link was found, when every selected row was rescued"
    )
    assert t.fit_result.fit_metrics["group_totals_before"] == {"a": 7, "b": 3}
    assert t.fit_result.fit_metrics["group_totals_after"] == {"a": 7, "b": 3}
    assert t.fit_result.fit_metrics["n_rows_removed"] == 0
    assert any("EVERY row of protected group" in m for m in messages)


def test_tomek_says_when_cleaning_left_a_group_unmeasurable():
    """Measured before: {'a': 6, 'b': 2} became {'a': 5, 'b': 1} with
    `fit_result.warnings == []`; a one-row group is a could-not-check for every
    per-group statistic computed afterwards."""
    X = pd.DataFrame(
        {
            "f0": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0, 10.05],
            "f1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 10.0, 10.05],
            "group": ["a"] * 6 + ["b"] * 2,
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    t = TomekResampler(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, _), messages = _caught(lambda: t.get_resampled_data(X, y))

    after = Counter(X_res["group"])
    assert after["b"] == 1, f"fixture no longer strips group b down: {dict(after)}"
    warned = _joined(t.fit_result.warnings)
    assert "fewer than two rows" in warned, f"a one-row group was not disclosed: {warned!r}"
    assert "could-not-check" in warned
    assert any("unevenly across" in m for m in messages)


def test_control_a_healthy_tomek_run_cleans_and_says_nothing():
    """Over-correction control: proportionate cleaning on a frame where every
    group survives must not warn at all, or the disclosures above are noise."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "group": ["a"] * 120 + ["b"] * 80,
        }
    )
    y = (rng.random(n) < 0.5).astype(int)
    t = TomekResampler(protected_attributes=["group"], random_state=42).fit(X, y)
    (X_res, y_res), messages = _caught(lambda: t.get_resampled_data(X, y))

    assert len(X_res) < n, "the cleaning stopped cleaning"
    assert len(X_res) == len(y_res)
    assert set(X_res["group"]) == {"a", "b"}
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"
    assert messages == []
    # The measurement is on the result object whether or not anything is wrong.
    assert t.fit_result.fit_metrics["group_totals_before"] == {"a": 120, "b": 80}
    assert t.fit_result.fit_metrics["n_rows_removed"] == n - len(X_res)


# ===========================================================================
# 5. CounterfactualAugmenter: twinnability was computed per ATTRIBUTE while the
#    swap moves the JOINT group.
# ===========================================================================


def test_twinnability_is_measured_on_the_joint_group():
    """Measured before: fit reported `n_twins_possible: 0` and warned "no
    counterfactual twin exists ... get_resampled_data() will return the data
    UNCHANGED", and get_resampled_data then returned 100 -> 200 rows and wrote
    `n_twins: 100` into the same dict."""
    df = pd.DataFrame(
        {"race": ["A"] * 60 + ["B"] * 40, "gender": ["M"] * 100, "f": np.arange(100.0)}
    )
    y = np.zeros(100, dtype=int)
    a = CounterfactualAugmenter(protected_attributes=["race", "gender"])
    _, fit_messages = _caught(lambda: a.fit(df, y))

    metrics = a.fit_result.fit_metrics
    assert metrics["levels_per_attribute"] == {"race": 2, "gender": 1}, "fixture drifted"
    assert metrics["n_joint_groups"] == 2
    assert metrics["n_twins_possible"] == 100, (
        f"a frame that twins every row reported n_twins_possible={metrics['n_twins_possible']}"
    )
    warned = _joined(a.fit_result.warnings)
    assert "no counterfactual twin exists" not in warned, (
        f"fit still claims no twin exists for a frame that twins every row: {warned!r}"
    )
    assert "partly counterfactual" in warned, (
        "the single-level attribute is a real fact and must still be reported"
    )

    X_res, y_res = a.get_resampled_data(df, y)
    assert len(X_res) == 200 and len(y_res) == 200
    assert a.fit_result.fit_metrics["n_twins"] == 100
    assert a.fit_result.fit_metrics["n_twins"] <= a.fit_result.fit_metrics["n_twins_possible"], (
        "the promise and the outcome still contradict each other in the same dict"
    )
    assert Counter(map(tuple, X_res[["race", "gender"]].values)) == Counter(
        {("A", "M"): 100, ("B", "M"): 100}
    )


def test_repeated_calls_do_not_stack_the_same_disclosure():
    """Three calls used to leave four entries on fit_result.warnings, so a
    reader counting warnings saw a worsening run where nothing changed."""
    df = pd.DataFrame({"g": ["A"] * 20, "f": np.arange(20.0)})
    y = np.zeros(20, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = CounterfactualAugmenter(protected_attributes=["g"]).fit(df, y)
        for _ in range(3):
            a.get_resampled_data(df, y)

    warned = a.fit_result.warnings
    assert len(warned) == len(set(warned)), f"the same sentence was recorded twice: {warned}"
    assert sum("no counterfactual twins were created" in w for w in warned) == 1
    assert sum("no counterfactual twin exists" in w for w in warned) == 1


def test_control_a_single_level_attribute_still_says_there_is_no_twin():
    """Over-correction control for the joint-group rule: the case the stage 2
    fix was FOR must keep refusing."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"gender": ["F"] * 60, "score": rng.normal(size=60)})
    y = rng.integers(0, 2, 60)
    a = CounterfactualAugmenter(protected_attributes=["gender"])
    with pytest.warns(UserWarning, match="no counterfactual twin exists"):
        a.fit(df, y)

    assert a.fit_result.fit_metrics["n_twins_possible"] == 0
    assert a.fit_result.fit_metrics["n_joint_groups"] == 1
    X_aug, _ = a.get_resampled_data(df, y)
    assert len(X_aug) == 60
    assert "no counterfactual twins were created" in _joined(a.fit_result.warnings)


def test_control_two_levels_still_double_the_frame_in_silence():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"gender": ["F"] * 30 + ["M"] * 30, "score": rng.normal(size=60)})
    y = rng.integers(0, 2, 60)
    a = CounterfactualAugmenter(protected_attributes=["gender"])
    (_, messages) = _caught(lambda: a.fit(df, y))
    X_aug, y_aug = a.get_resampled_data(df, y)

    assert len(X_aug) == 120 and len(y_aug) == 120
    assert a.fit_result.fit_metrics["n_twins"] == 60
    assert a.fit_result.fit_metrics["n_joint_groups"] == 2
    assert a.fit_result.warnings == [], f"the working run warned: {a.fit_result.warnings}"
    assert messages == []
    assert Counter(X_aug["gender"]) == Counter({"F": 60, "M": 60})


# ===========================================================================
# 6. Resampler (transformers.py): the early return that kept the LAST run.
# ===========================================================================


def _resampler_frame(seed: int = 4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f": rng.normal(size=60), "grp": ["a"] * 59 + ["b"]})
    y = np.array([i % 2 for i in range(59)] + [1])
    return X, y


def test_a_zero_row_resample_clears_the_previous_runs_numbers():
    """Measured before: after a 90-row resample, a 0-row call left
    `n_rows_out=90`, `group_totals_after={'a': 60, 'b': 30}` and three warnings
    standing, every one of them about a frame that was not the one handed in."""
    X, y = _resampler_frame()
    r = Resampler(protected_attributes=["grp"]).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.get_resampled_data(X, y)
    assert r.resample_result["n_rows_out"] > 0, "fixture no longer records a first run"
    assert r.fit_result.warnings, "fixture no longer produces disclosures to go stale"

    empty = X.iloc[0:0]
    (X_res, y_res), messages = _caught(lambda: r.get_resampled_data(empty, np.array([])))

    assert len(X_res) == 0
    assert r.resample_result["n_rows_out"] == 0, (
        f"the object still reports the previous run: {r.resample_result['n_rows_out']}"
    )
    assert r.resample_result["group_totals_after"] == {}
    assert r.resample_result["max_replication_factor"] is None
    assert r.resample_result["unreachable_cells"] is None
    assert "could_not_check" in r.resample_result
    warned = _joined(r.fit_result.warnings)
    assert "nothing was resampled" in warned
    assert "60" not in warned, f"a stale disclosure about the previous frame survived: {warned!r}"
    assert any("nothing was resampled" in m for m in messages)


def test_a_single_group_frame_says_between_group_balance_was_not_measured():
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"f": rng.normal(size=40), "grp": ["a"] * 40})
    y = np.array([0, 1] * 20)
    r = Resampler(protected_attributes=["grp"]).fit(X, y)
    (X_res, _), messages = _caught(lambda: r.get_resampled_data(X, y))

    assert len(X_res) == 40
    assert r.resample_result["unreachable_cells"] == [], "fixture drifted"
    warned = _joined(r.fit_result.warnings)
    assert "between-group balance was NOT measured" in warned, (
        f"a one-group frame reported a balanced result in silence: {warned!r}"
    )
    assert "only one protected group" in warned


def test_control_a_healthy_resample_still_records_what_it_did():
    X, y = _healthy_cells_frame()
    r = Resampler(protected_attributes=["grp"], balance_by="group_label").fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_res, y_res = r.get_resampled_data(X, y)

    counts = Counter(X_res["grp"])
    assert len(set(counts.values())) == 1, f"groups came back uneven: {dict(counts)}"
    assert len(X_res) == len(y_res) == 500
    assert r.resample_result["n_rows_out"] == 500
    assert r.resample_result["unreachable_cells"] == []
    assert r.resample_result["group_totals_after"] == {"A": 250, "B": 250}
    assert "between-group balance was NOT measured" not in _joined(r.fit_result.warnings)
