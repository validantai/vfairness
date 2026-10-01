"""Beta Go-Live stage 2, group s2g06: data_balancing must not grade what it never measured.

Nine defects were reproduced at the public API in
``preprocessing/feature_engineering/data_balancing.py``. Every one of them had
the same shape: an input the technique cannot run on produced a CONFIDENT
value instead of a refusal, and in four cases the confident value graded the
failure BETTER than the success.

Each test below asserts the three states at the PUBLIC entry point (the
transformer method a caller calls, not a private helper), and each is paired
with a control proving healthy data still produces the same measurement it
always did. A fix that makes everything refuse is a worse defect than the one
it replaces, and it passes any test that only checks the degenerate case.

Measured before the fix (2026-09-16), so the pins below have a real anchor:

* ``InversePropensityWeighter.fit`` on a ONE-level attribute reported
  ``max_dev_after = 0.0`` (perfect uniform representation) and 40
  ``propensity_scores`` of exactly 1.0, for a model that was never fitted.
* One ``np.inf`` cell made the solver raise; a bare ``except`` swallowed it and
  fell back to the marginal group frequency, which drives the weighted shares
  to EXACTLY uniform. The broken run reported ``max_dev_after = 0.000000``, the
  working model ``0.119828``: the failure graded better than the success, with
  ``warnings == []`` on both.
* ``get_sample_weights`` refitted from scratch on whatever batch it was handed,
  so a row's weight depended on its batch mates (1.0672 in the fit frame,
  1.2341 alone) and a single-group batch came back as thirty weights of 1.0.
* SMOTE / ADASYN filled a (group, label) cell holding ONE row to 30 rows, all
  30 byte-identical copies, while the only warning present said "Every cell
  that does exist was balanced to 30 rows".
* SMOTE's synthesis matrix was ``.fillna(0.0)``, so a column nobody measured
  came back with a group mean of exactly 0.000 and 30 rows outside the convex
  range of the real data.
* ``CounterfactualAugmenter.fit`` on a one-level attribute returned a result
  byte-identical to the working two-level case and then no-opped 60 rows in /
  60 rows out, silently.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vfairness import (
    ADASYNResampler,
    CounterfactualAugmenter,
    InversePropensityWeighter,
    PropensityScoreWeighter,
    SMOTEResampler,
)

# ---------------------------------------------------------------------------
# Fixtures. Each one states which branch it is there to exercise, because a
# fixture that never reaches the branch makes a sabotage look green.
# ---------------------------------------------------------------------------


def _two_group_frame(n_a: int = 80, n_b: int = 20, seed: int = 11):
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


def _single_group_frame(n: int = 40, seed: int = 11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"grp": ["A"] * n, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})


def _one_row_cell_frame(seed: int = 4):
    """59 rows in group a, ONE row in group b: the (b, label=1) cell holds a
    single real observation, so there is nothing to interpolate from."""
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


def _balanced_cells_frame(n_a: int = 250, n_b: int = 50, seed: int = 0):
    """Every (group, label) cell is populated and none is degenerate: the
    healthy control for the resamplers."""
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


def _unmeasured_column_frame(seed: int = 1):
    """Group b's f0 was never measured. f1 is measured for every row, so the
    neighbour search still has something to work with."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": np.concatenate([rng.normal(50, 5, 40), np.full(10, np.nan)]),
            "f1": rng.normal(size=50),
        }
    )
    X["group"] = ["a"] * 40 + ["b"] * 10
    y = np.array([0] * 20 + [1] * 20 + [0] * 5 + [1] * 5)
    return X, y


def _joined(messages) -> str:
    return " ".join(messages)


# ===========================================================================
# Finding 1: InversePropensityWeighter.fit, degenerate single-group branch
# ===========================================================================


def test_ipw_on_one_group_does_not_grade_itself_perfectly_balanced():
    df = _single_group_frame()
    assert df["grp"].nunique() == 1, "fixture no longer reaches the single-group branch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)

    balance = t.fit_result.fit_metrics["group_balance"]
    assert balance["n_groups"] == 1

    # THREE STATES. 0.0 here reads as "perfect uniform representation" to
    # anything that grades on it. There is no second group, so there is no
    # between-group deviation to measure at all.
    assert balance["max_dev_after"] is None, (
        f"a run with no comparison group graded itself {balance['max_dev_after']!r}"
    )
    assert balance["share_after"] is None

    scores = np.asarray(t._weights_result["propensity_scores"], dtype=float)
    assert np.isnan(scores).all(), (
        "propensity scores were reported for a model that was never fitted: "
        f"{sorted(set(scores.tolist()))[:3]}"
    )

    assert t.fit_result.fit_metrics["propensity_model"] == "single_group"
    warned = _joined(t.fit_result.warnings)
    assert "no propensity model was estimated" in warned
    assert "one observed group" in warned
    assert any("no propensity model was estimated" in str(w.message) for w in caught), (
        "the refusal never reached a Python warning"
    )

    # max_dev_BEFORE is a fact about the rows that were read and must survive.
    assert balance["max_dev_before"] == 0.0


def test_ipw_control_two_groups_still_measures_the_same_numbers():
    """The healthy path must be bit-for-bit what it was before the refusal."""
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)

    balance = t.fit_result.fit_metrics["group_balance"]
    assert t.fit_result.fit_metrics["propensity_model"] == "fitted"
    assert balance["max_dev_before"] == pytest.approx(0.30, abs=1e-9)
    assert balance["max_dev_after"] == pytest.approx(0.119828, abs=1e-6)
    assert balance["share_after"]["A"] == pytest.approx(0.6198278, abs=1e-6)
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"

    scores = np.asarray(t._weights_result["propensity_scores"], dtype=float)
    assert np.isfinite(scores).all() and len(set(scores.tolist())) > 50, (
        "a fitted model must produce a per-row propensity, not a handful of literals"
    )


# ===========================================================================
# Finding 2: InversePropensityWeighter.fit, the swallowed solver failure
# ===========================================================================


def test_ipw_solver_failure_is_not_reported_as_a_better_balance():
    healthy = _two_group_frame()
    broken = healthy.copy()
    broken.loc[0, "x1"] = np.inf  # the solver refuses this frame

    good = InversePropensityWeighter(protected_attributes=["grp"]).fit(healthy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad = InversePropensityWeighter(protected_attributes=["grp"]).fit(broken)

    # The fixture really does hit the solver, not some earlier guard.
    assert bad.fit_result.fit_metrics["propensity_model"] == "solver_failed"
    assert good.fit_result.fit_metrics["propensity_model"] == "fitted"

    good_dev = good.fit_result.fit_metrics["group_balance"]["max_dev_after"]
    bad_dev = bad.fit_result.fit_metrics["group_balance"]["max_dev_after"]
    assert good_dev == pytest.approx(0.119828, abs=1e-6)
    assert bad_dev is None, (
        f"the failed fit graded itself {bad_dev!r}; before the fix this was 0.0, "
        f"which is a BETTER score than the working model's {good_dev:.6f}"
    )

    weights = np.asarray(bad._weights_result["sample_weights"], dtype=float)
    assert np.isnan(weights).all(), (
        "a weight that was never derived must be NaN, not a trainable number: "
        f"{sorted(set(weights.tolist()))[:3]}"
    )
    scores = np.asarray(bad._weights_result["propensity_scores"], dtype=float)
    assert np.isnan(scores).all(), (
        "the marginal group frequency was returned under the name 'propensity_scores'"
    )

    warned = _joined(bad.fit_result.warnings)
    assert "solver refused" in warned and "ValueError" in warned, (
        f"the swallowed exception is still invisible: {warned!r}"
    )
    assert any("solver refused" in str(w.message) for w in caught)

    # And the imbalance that WAS measured is still reported.
    assert bad.fit_result.fit_metrics["group_balance"]["max_dev_before"] == pytest.approx(0.30)


def test_ipw_control_the_weights_still_move_the_group_shares():
    """Over-correction control: the working model must keep working."""
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)
    weights = np.asarray(t._weights_result["sample_weights"], dtype=float)

    assert np.isfinite(weights).all() and (weights > 0).all()
    assert len(set(weights.tolist())) == 83, "the fitted model no longer gives a per-row weight"
    balance = t.fit_result.fit_metrics["group_balance"]
    assert balance["max_dev_after"] < balance["max_dev_before"], "reweighting stopped helping"


# ===========================================================================
# Finding 3: InversePropensityWeighter.get_sample_weights
# ===========================================================================


def test_ipw_sample_weights_are_a_property_of_the_row_not_of_the_batch():
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)

    rows = [0, 1, 2, 80, 81]
    in_frame = t.get_sample_weights(df)[rows]
    standalone = t.get_sample_weights(df.iloc[rows])
    np.testing.assert_allclose(
        standalone,
        in_frame,
        rtol=0,
        atol=1e-9,
        err_msg="the same rows got different weights depending on their batch mates",
    )

    # A batch holding one group used to take the degenerate branch and come back
    # as the neutral 1.0 for every row.
    single_group_batch = df[df["grp"] == "A"].head(30)
    assert single_group_batch["grp"].nunique() == 1, "fixture is not single-group"
    batch_weights = t.get_sample_weights(single_group_batch)
    assert not np.allclose(batch_weights, 1.0), (
        "a single-group batch came back as the neutral 'no reweighting needed' value"
    )
    np.testing.assert_allclose(batch_weights, t.get_sample_weights(df)[:30], rtol=0, atol=1e-9)

    one_b_row = t.get_sample_weights(df.iloc[[80]])
    assert one_b_row[0] == pytest.approx(4.41485972, abs=1e-6), (
        f"the minority row's fitted weight was replaced by {one_b_row[0]!r}"
    )


def test_ipw_sample_weights_refuse_a_group_the_model_never_saw():
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)

    unseen = df.iloc[[0, 1]].copy()
    unseen.loc[:, "grp"] = "C"
    with pytest.warns(UserWarning, match="no propensity was fitted for group"):
        weights = t.get_sample_weights(unseen)
    assert np.isnan(weights).all(), f"an unfitted group got a real-looking weight: {weights}"


def test_ipw_sample_weights_refuse_when_no_model_was_estimated():
    t = InversePropensityWeighter(protected_attributes=["grp"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t.fit(_single_group_frame())

    with pytest.warns(UserWarning, match="no propensity model was estimated"):
        weights = t.get_sample_weights(_single_group_frame())
    # With one group the identity IS the complete answer, so 1.0 is correct
    # here; what must not happen is it arriving silently.
    assert np.allclose(weights, 1.0)
    assert t.fit_result.fit_metrics["group_balance"]["max_dev_after"] is None


def test_ipw_control_get_sample_weights_reproduces_the_fit_frame_weights():
    df = _two_group_frame()
    t = InversePropensityWeighter(protected_attributes=["grp"]).fit(df)
    np.testing.assert_allclose(
        t.get_sample_weights(df),
        np.asarray(t._weights_result["sample_weights"], dtype=float),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("cls", [PropensityScoreWeighter, InversePropensityWeighter])
def test_control_both_weighters_still_return_one_positive_weight_per_row(cls):
    df = _two_group_frame()
    w = cls(protected_attributes=["grp"]).fit(df).get_sample_weights(df)
    assert len(w) == len(df)
    assert np.isfinite(w).all() and (w > 0).all()
    assert not np.allclose(w, w[0]), "the reweighting became a no-op"


def test_control_psw_stabilised_scale_is_the_fit_time_scale():
    """mean-1 normalisation must use the FIT frame's mean, or a batch rescales
    itself and the same row gets two different stabilised weights."""
    df = _two_group_frame()
    t = PropensityScoreWeighter(protected_attributes=["grp"]).fit(df)
    full = t.get_sample_weights(df)
    assert full.mean() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        t.get_sample_weights(df.iloc[[0, 1, 2, 80, 81]]),
        full[[0, 1, 2, 80, 81]],
        rtol=0,
        atol=1e-9,
    )


# ===========================================================================
# Findings 4 and 5: SMOTE / ADASYN fill a one-row cell by duplication
# ===========================================================================


@pytest.mark.parametrize("cls", [ADASYNResampler, SMOTEResampler])
def test_a_cell_filled_by_duplication_says_so(cls):
    df, y = _one_row_cell_frame()
    assert (df["group"] == "b").sum() == 1, "fixture no longer has a one-row cell"

    t = cls(protected_attributes=["group"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_res, _ = t.get_resampled_data(df, y)

    b_rows = X_res[X_res["group"] == "b"]
    assert len(b_rows) == 30, "fixture no longer reaches the oversampling path"
    assert len(b_rows[["f0", "f1"]].drop_duplicates()) == 1, (
        "fixture changed: the cell is no longer filled by duplication"
    )

    warned = _joined(t.fit_result.warnings)
    assert "DUPLICATION" in warned, (
        f"30 byte-identical copies were returned as resampled data, unreported: {warned!r}"
    )
    assert "label=" in warned and "(b," in warned, "the warning does not name the cell"
    assert "effective sample size is unchanged" in warned
    assert any("DUPLICATION" in str(w.message) for w in caught)

    # The pre-existing absent-cell warning must survive, and must no longer end
    # in a sentence that contradicts the duplication it sits next to.
    assert "not group-balanced" in warned
    assert "Every cell that does exist was balanced to 30 rows." not in warned, (
        "the misleading closing sentence is still shipped"
    )


@pytest.mark.parametrize("cls", [ADASYNResampler, SMOTEResampler])
def test_control_a_healthy_resample_warns_about_nothing_and_still_balances(cls):
    """A warning on every run is a warning on none."""
    X, y = _balanced_cells_frame()
    t = cls(protected_attributes=["grp"]).fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_res, y_res = t.get_resampled_data(X, y)

    counts = Counter(X_res["grp"])
    assert len(set(counts.values())) == 1, f"groups came back uneven: {dict(counts)}"
    assert counts["B"] > Counter(X["grp"])["B"], "the minority group did not grow"
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"
    assert [str(w.message) for w in caught] == []
    assert len(X_res) == len(y_res)

    # And the synthesis is real synthesis: the new rows are not copies.
    b_rows = X_res[X_res["grp"] == "B"]
    assert len(b_rows[["f1", "f2"]].drop_duplicates()) > 200


# ===========================================================================
# Finding 6: SMOTE wrote 0.0 into a column nobody measured
# ===========================================================================


def test_smote_leaves_an_unmeasured_feature_unmeasured():
    X, y = _unmeasured_column_frame()
    assert X.loc[X["group"] == "b", "f0"].isna().all(), "fixture: b's f0 must be unmeasured"
    measured = X["f0"].dropna()

    t = SMOTEResampler(protected_attributes=["group"]).fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_res, _ = t.get_resampled_data(X, y)

    assert len(X_res) > len(X), "fixture no longer reaches the synthesis path"

    # Before: 30 output rows carried f0 == 0.0 and the group mean read 0.000.
    assert int((X_res["f0"] == 0.0).sum()) == 0, "a fabricated zero is still written out"
    b_mean = X_res.groupby("group")["f0"].mean()["b"]
    assert np.isnan(b_mean), f"group b reported a mean of {b_mean!r} for a column nobody measured"

    # The class docstring's promise: a synthetic sample lies in the convex range
    # of the real samples of its cell.
    finite = X_res["f0"].dropna()
    assert finite.min() >= measured.min() - 1e-9 and finite.max() <= measured.max() + 1e-9, (
        "a synthetic value fell outside the convex range of the measured data"
    )

    warned = _joined(t.fit_result.warnings)
    assert "never measured" in warned and "'f0'" in warned
    assert "carry NaN" in warned
    assert any("never measured" in str(w.message) for w in caught)


def test_control_smote_still_synthesises_real_numbers_when_nothing_is_missing():
    rng = np.random.default_rng(99)  # not the fixture's seed, or the draws repeat
    X, y = _unmeasured_column_frame()
    X = X.copy()
    # The caller measures f0 for group b too, so nothing is missing anywhere.
    X.loc[X["f0"].isna(), "f0"] = rng.normal(50, 5, int(X["f0"].isna().sum()))
    measured = X["f0"]
    assert measured.notna().all() and measured.nunique() == len(X)

    t = SMOTEResampler(protected_attributes=["group"]).fit(X, y)
    X_res, _ = t.get_resampled_data(X, y)

    assert len(X_res) > len(X)
    assert X_res["f0"].notna().all(), "healthy data came back with holes in it"
    assert X_res["f0"].min() >= measured.min() - 1e-9
    assert X_res["f0"].max() <= measured.max() + 1e-9
    assert t.fit_result.warnings == [], f"the healthy run warned: {t.fit_result.warnings}"
    assert X_res[X_res["group"] == "b"]["f0"].nunique() > 1, "synthesis produced copies"


# ===========================================================================
# Finding 7: PropensityScoreWeighter.fit
# ===========================================================================


def test_psw_on_one_group_reports_that_nothing_was_estimated():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
    X["group"] = ["a"] * 60

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = PropensityScoreWeighter(protected_attributes=["group"]).fit(X)

    balance = r.fit_result.fit_metrics["group_balance"]
    assert balance["max_dev_after"] is None
    assert r.fit_result.fit_metrics["propensity_model"] == "single_group"
    scores = np.asarray(r._weights_result["propensity_scores"], dtype=float)
    assert np.isnan(scores).all(), "60 literal 1.0s were reported as estimated propensities"
    assert r.fit_result.warnings, "the degenerate fit still reports a warning-free success"
    assert any("no propensity model" in str(w.message) for w in caught)


def test_psw_with_no_measured_feature_variation_refuses_rather_than_scoring():
    """Two real groups, but every feature unreadable: the design matrix collapses
    and 'propensity' degenerates to the marginal group frequency P(group)."""
    X = pd.DataFrame({"f1": np.full(60, np.nan), "f2": np.full(60, np.nan)})
    X["group"] = ["a"] * 30 + ["b"] * 30

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = PropensityScoreWeighter(protected_attributes=["group"]).fit(X)

    assert r.fit_result.fit_metrics["propensity_model"] == "not_estimable"
    assert r.fit_result.fit_metrics["group_balance"]["max_dev_after"] is None
    assert np.isnan(np.asarray(r._weights_result["propensity_scores"], dtype=float)).all()
    assert np.isnan(np.asarray(r._weights_result["sample_weights"], dtype=float)).all(), (
        "weights derived from no model must be NaN, not 1.0"
    )
    warned = _joined(r.fit_result.warnings)
    assert "no feature column has any measured variation" in warned
    assert any(
        "not estimable" in str(w.message) or "no propensity model" in str(w.message) for w in caught
    )

    # A constant column is not "variation" either, and np.ptp is used rather
    # than a variance compared against exact zero.
    const = pd.DataFrame({"f1": np.full(60, 0.9), "group": ["a"] * 30 + ["b"] * 30})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = PropensityScoreWeighter(protected_attributes=["group"]).fit(const)
    assert r2.fit_result.fit_metrics["propensity_model"] == "not_estimable"


def test_control_psw_two_groups_with_real_features_still_fits_and_improves():
    df = _two_group_frame()
    r = PropensityScoreWeighter(protected_attributes=["grp"]).fit(df)
    balance = r.fit_result.fit_metrics["group_balance"]
    assert r.fit_result.fit_metrics["propensity_model"] == "fitted"
    assert balance["max_dev_after"] == pytest.approx(0.119828, abs=1e-6)
    assert balance["max_dev_after"] < balance["max_dev_before"]
    scores = np.asarray(r._weights_result["propensity_scores"], dtype=float)
    assert np.isfinite(scores).all() and len(set(scores.tolist())) > 50
    assert r.fit_result.warnings == []


def test_psw_records_how_much_of_the_design_matrix_was_imputed():
    """An imputation that the estimate depends on has to be visible."""
    df = _two_group_frame()
    df = df.copy()
    df.loc[0:4, "x2"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = PropensityScoreWeighter(protected_attributes=["grp"]).fit(df)
    assert r.fit_result.fit_metrics["propensity_model"] == "fitted"
    assert r.fit_result.fit_metrics["n_imputed_feature_cells"] == 5
    assert any("entered the design matrix as 0.0" in str(w.message) for w in caught)


# ===========================================================================
# Findings 8 and 9: CounterfactualAugmenter
# ===========================================================================


def test_counterfactual_fit_says_when_no_twin_can_exist():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"gender": ["F"] * 60, "score": rng.normal(size=60)})
    y = rng.integers(0, 2, 60)
    assert df["gender"].nunique() == 1, "fixture no longer reaches the one-level branch"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = CounterfactualAugmenter(protected_attributes=["gender"]).fit(df, y)

    assert a.fit_result.fit_metrics["levels_per_attribute"] == {"gender": 1}
    assert a.fit_result.fit_metrics["n_twins_possible"] == 0
    warned = _joined(a.fit_result.warnings)
    assert "no counterfactual twin exists" in warned, (
        "fit() reported a warning-free success for a mitigation that cannot run"
    )
    assert any("no counterfactual twin exists" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        X_aug, y_aug = a.get_resampled_data(df, y)
    assert len(X_aug) == 60, "the no-op is expected; the silence is not"
    assert a.fit_result.fit_metrics["n_twins"] == 0
    assert "no counterfactual twins were created" in _joined(a.fit_result.warnings)
    assert any("NO counterfactual twins" in str(w.message) for w in caught2)


def test_counterfactual_fit_refuses_an_attribute_that_is_not_a_column():
    """It used to accept, report success, and fail later with a KeyError out of
    get_resampled_data."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(30, 3))
    with pytest.raises(ValueError, match="Protected attributes not found"):
        CounterfactualAugmenter(protected_attributes=["gender"]).fit(arr, rng.integers(0, 2, 30))


def test_counterfactual_does_not_invent_a_group_for_a_row_that_has_none():
    """50 rows labelled 'A' and 10 with no recorded race used to come back as 70
    rows, because each unmeasured row was handed a twin with race='A'."""
    df = pd.DataFrame(
        {"race": ["A"] * 50 + [np.nan] * 10, "f": np.zeros(60), "y": np.zeros(60, int)}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = CounterfactualAugmenter(protected_attributes=["race"]).fit(df, df["y"])
        X_aug, _ = a.get_resampled_data(df, df["y"])

    assert len(X_aug) == 60, f"{len(X_aug) - 60} twin(s) were invented for unmeasured rows"
    assert X_aug["race"].isna().sum() == 10, "an unmeasured group was filled in"
    assert a.fit_result.fit_metrics["n_rows_without_a_recorded_group"] == 10
    warned = _joined(a.fit_result.warnings)
    assert "no recorded value" in warned
    assert any("no recorded value" in str(w.message) for w in caught)


def test_control_counterfactual_two_levels_still_doubles_the_frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"gender": ["F"] * 30 + ["M"] * 30, "score": rng.normal(size=60)})
    y = rng.integers(0, 2, 60)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = CounterfactualAugmenter(protected_attributes=["gender"]).fit(df, y)
        X_aug, y_aug = a.get_resampled_data(df, y)

    assert len(X_aug) == 120 and len(y_aug) == 120
    assert a.fit_result.fit_metrics["n_twins"] == 60
    assert a.fit_result.fit_metrics["levels_per_attribute"] == {"gender": 2}
    assert a.fit_result.warnings == [], f"the working run warned: {a.fit_result.warnings}"
    assert [str(w.message) for w in caught] == []
    assert Counter(X_aug["gender"]) == Counter({"F": 60, "M": 60})


def test_control_counterfactual_three_levels_still_balances_toward_uniform():
    """The 2026-09-10 running-count allocation must survive untouched."""
    df = pd.DataFrame({"race": ["A"] * 80 + ["B"] * 15 + ["C"] * 5})
    y = np.zeros(100, dtype=int)
    a = CounterfactualAugmenter(protected_attributes=["race"]).fit(df, y)
    X_aug, _ = a.get_resampled_data(df, y)
    assert Counter(X_aug["race"]) == Counter({"A": 80, "C": 65, "B": 55})


def test_control_counterfactual_multi_attribute_branch_is_unchanged():
    df = pd.DataFrame(
        {
            "race": ["A"] * 70 + ["B"] * 30,
            "gender": ["M"] * 45 + ["F"] * 25 + ["M"] * 22 + ["F"] * 8,
        }
    )
    y = np.zeros(100, dtype=int)
    a = CounterfactualAugmenter(protected_attributes=["race", "gender"]).fit(df, y)
    X_aug, _ = a.get_resampled_data(df, y)
    after = Counter(map(tuple, X_aug[["race", "gender"]].values))
    assert after == Counter({("A", "F"): 51, ("A", "M"): 50, ("B", "M"): 50, ("B", "F"): 49})
    assert a.fit_result.warnings == []
