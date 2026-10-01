"""Beta Go-Live stage 2, group s2g00: feature-engineering transformers.

Every test here pins a value that was FABRICATED before 2026-09-16: a statistic
that is 0.0 / True / 2 / n by construction rather than by measurement, handed
back at the public entry point with an empty warnings list, indistinguishable
from the same object produced by a genuine run.

The shape of every pin is the same:

  * the DEGENERATE case must reach a third state (refused, omitted, named,
    counted or raised), never a neutral default, and
  * a CONTROL on healthy data in the same test must still produce the correct
    measurement. A fix that makes everything refuse is a worse defect than the
    one it replaces, and it passes any test that only checks the degenerate half.

Several tests also assert that the FIXTURE reaches the branch under test (a
constant score vector, a 1-row intersection with a NaN std, a genuinely absent
(group, label) cell), because a pin that never enters the code it names is green
for the wrong reason.

Covers: resampling, fair_representation, feature_suppression,
disparate_impact_removal, label_massaging, residual_transform,
correlation_reduction, intersectional_transform.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness import (
    CorrelationReducer,
    DisparateImpactRemover,
    FeatureSuppressor,
    IntersectionalTransformer,
    LabelMassager,
    Resampler,
    ResidualTransformer,
)


def _messages(recorded) -> list[str]:
    return [str(w.message) for w in recorded]


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def test_resampler_discloses_replication_and_unreachable_cells():
    """F1: 30 rows copied from ONE observation, returned as a balanced frame.

    Before: get_resampled_data returned (DataFrame 90 rows, ndarray 90) with
    fit_result.warnings == [] and fit_metrics == {'strategy', 'balance_by'}.
    Group b was 30 rows derived from exactly 1 distinct observation and the
    requested (group, label) balance was unreachable, both silently.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"f0": rng.normal(size=60), "f1": rng.normal(size=60), "group": ["a"] * 59 + ["b"]}
    )
    y = np.array([i % 2 for i in range(59)] + [0])

    r = Resampler(
        protected_attributes=["group"], strategy="oversample", balance_by="group_label"
    ).fit(df, y)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Xb, yb = r.get_resampled_data(df, y)

    # The fixture really is the degenerate case: one cell of one row, and the
    # (b, label=1) cell does not exist at all.
    b_rows = Xb[Xb["group"] == "b"]
    assert len(b_rows) == 30 and len(b_rows.drop_duplicates()) == 1

    res = r.resample_result
    assert res["max_replication_factor"] == 30.0
    assert res["n_rows_out"] == 90 and res["n_effective_rows"] == 60
    assert res["n_duplicated_rows"] == 30
    assert res["unreachable_cells"] == ["(b, label=1)"]

    warned = r.fit_result.warnings
    assert any("no examples to resample from" in m for m in warned)
    assert any("are copies, not observations" in m for m in warned)
    assert any("Resampler:" in m for m in _messages(w))

    # CONTROL: a genuinely balanceable frame reports no unreachable cell, no
    # over-replication, and comes back actually balanced.
    gh = ["a"] * 30 + ["b"] * 30
    yh = np.array(([0, 1] * 15) + ([0, 1] * 15))
    dh = pd.DataFrame({"f0": rng.normal(size=60), "f1": rng.normal(size=60), "group": gh})
    rh = Resampler(
        protected_attributes=["group"], strategy="oversample", balance_by="group_label"
    ).fit(dh, yh)
    Xh, _ = rh.get_resampled_data(dh, yh)
    assert rh.resample_result["unreachable_cells"] == []
    assert rh.resample_result["max_replication_factor"] == 1.0
    assert rh.resample_result["n_duplicated_rows"] == 0
    assert Xh["group"].value_counts().to_dict() == {"a": 30, "b": 30}
    assert rh.fit_result.warnings == []


def test_resampler_fit_result_says_the_group_totals_came_out_uneven():
    """F15: {'a': 76, 'b': 38} returned by a balancer, warnings == []."""
    rng = np.random.default_rng(5)
    g = ["a"] * 59 + ["b"]
    y = np.array([0] * 38 + [1] * 21 + [1])
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60), "grp": g})

    r = Resampler(protected_attributes=["grp"]).fit(X, y)
    Xb, _ = r.get_resampled_data(X, y)

    assert Xb["grp"].value_counts().to_dict() == {"a": 76, "b": 38}
    assert r.resample_result["group_totals_after"] == {"a": 76, "b": 38}
    assert any("group totals after resampling are uneven" in m for m in r.fit_result.warnings)
    # The serialised report carries it too, not only the live object.
    assert r.fit_result.to_dict()["warnings"]


# ---------------------------------------------------------------------------
# fair_representation  (torch-only)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from vfairness import FairRepresentationTransformer  # noqa: E402


def _fr_frame(levels: list[str], n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "grp": [levels[i % len(levels)] for i in range(n)],
            "income": rng.normal(54000, 13000, n),
            "age": rng.integers(20, 70, n).astype(float),
            "score": rng.random(n),
        }
    )


def test_fair_representation_refuses_to_fit_on_zero_rows():
    """F2: fit() on 0 rows returned is_fitted=True with n_groups=2 and produced
    an encoder whose transform of 60 REAL rows was 100% NaN, warnings == []."""
    empty = pd.DataFrame({"age": [], "income": [], "score": [], "gender": []})
    with pytest.raises(ValueError, match="at least one row"):
        FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=3, epochs=5
        ).fit(empty)

    # CONTROL: one row of real data is enough to fit, and the result is finite.
    one = pd.DataFrame({"age": [30.0], "income": [5e4], "score": [0.5], "gender": ["a"]})
    t = FairRepresentationTransformer(
        protected_attributes=["gender"], representation_dim=2, epochs=3
    ).fit(one)
    assert t.is_fitted
    assert np.isfinite(t.transform(one).values).all()


def test_fair_representation_reports_the_observed_group_count_not_the_floor():
    """F3: a column with ONE level published n_groups == 2, byte-identical to a
    genuine 2-group run, with warnings == []."""
    single = _fr_frame(["a"])
    with pytest.warns(UserWarning, match="fewer than the 2 an adversary needs"):
        t = FairRepresentationTransformer(
            protected_attributes=["grp"], representation_dim=3, epochs=5
        ).fit(single)

    assert single["grp"].nunique() == 1
    assert t.fit_result.fit_metrics["n_groups"] == 1
    # The architectural head width is reported separately and is NOT a count.
    assert t.fit_result.fit_metrics["adversary_output_dim"] == 2
    assert any("NO fairness constraint was applied" in m for m in t.fit_result.warnings)

    # CONTROL: two real levels still measure 2, with nothing to warn about.
    two = _fr_frame(["a", "b"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        th = FairRepresentationTransformer(
            protected_attributes=["grp"], representation_dim=3, epochs=5
        ).fit(two)
    assert th.fit_result.fit_metrics["n_groups"] == 2
    assert th.fit_result.warnings == []
    assert not [m for m in _messages(w) if "adversary" in m]
    # And the two records are no longer indistinguishable.
    assert th.fit_result.fit_metrics != t.fit_result.fit_metrics


def test_fair_representation_fit_transform_carries_the_same_refusals():
    """F5: fit_transform is the single-call path most callers use; it handed
    back a (60, 3) 'fair representation' claiming 2 groups for one level, and a
    (0, 3) frame from a NaN-fitted encoder."""
    single = _fr_frame(["a"])
    t = FairRepresentationTransformer(protected_attributes=["grp"], representation_dim=3, epochs=5)
    with pytest.warns(UserWarning, match="fewer than the 2 an adversary needs"):
        out = t.fit_transform(single)
    assert out.shape == (60, 3)
    assert t.fit_result.fit_metrics["n_groups"] == 1

    empty = pd.DataFrame({"age": [], "income": [], "score": [], "grp": []})
    with pytest.raises(ValueError, match="at least one row"):
        FairRepresentationTransformer(
            protected_attributes=["grp"], representation_dim=3, epochs=5
        ).fit_transform(empty)

    # CONTROL: the healthy single-call path is unchanged and finite.
    two = _fr_frame(["a", "b"])
    th = FairRepresentationTransformer(protected_attributes=["grp"], representation_dim=3, epochs=5)
    Z = th.fit_transform(two)
    assert Z.shape == (60, 3) and np.isfinite(Z.values).all()
    assert th.fit_result.fit_metrics["n_groups"] == 2


def test_fair_representation_transform_refuses_a_missing_fitted_feature():
    """F4: a DROPPED column and a present-but-NaN column produced the identical
    representation, because the missing column was invented as 0.0 - which is
    not even the fit-time mean (income entered at -4.4733 sd)."""
    df = _fr_frame(["a", "b"])
    t = FairRepresentationTransformer(
        protected_attributes=["grp"], representation_dim=2, epochs=5
    ).fit(df)

    with pytest.raises(ValueError, match=r"required by fit are absent.*income"):
        t.transform(df.drop(columns=["income"]))
    with pytest.raises(ValueError, match="required by fit are absent"):
        t.transform(df[["grp"]])

    # A NaN is imputed with the FIT-TIME mean and counted where a caller sees it.
    holed = df.copy()
    holed.loc[0, "income"] = np.nan
    with pytest.warns(UserWarning, match="imputed with the fit-time mean"):
        t.transform(holed)
    assert t.transform_result["n_values_imputed"] == 1
    assert t.transform_result["imputed_per_feature"]["income"] == 1

    # CONTROL: the complete frame imputes nothing, warns nothing, and the
    # representation of the untouched rows is unchanged.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        full = t.transform(df)
    assert t.transform_result["n_values_imputed"] == 0
    assert _messages(w) == []
    assert np.isfinite(full.values).all()
    # The imputed row moves toward the centre, not >4 sd into the tail.
    holed_out = t.transform(holed)
    assert abs(holed_out.values[0] - full.values[0]).max() < 1.0


# ---------------------------------------------------------------------------
# feature_suppression
# ---------------------------------------------------------------------------


def _suppressor_frame(levels: np.ndarray, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    two = np.array(["a", "b"] * 30)
    proxy = np.where(two == "a", 1.0, 5.0) + rng.normal(0, 0.2, 60)
    noise = rng.normal(size=60)
    return pd.DataFrame({"proxy": proxy, "noise": noise, "group": levels})


def test_feature_suppressor_omits_an_unmeasurable_attribute_instead_of_0():
    """F6: correlation_before == {'group': 0.0} for a single-level attribute,
    for which eta is 0.0 BY CONSTRUCTION. Byte-identical feature column, two
    levels, reports 0.5329."""
    single = _suppressor_frame(np.array(["a"] * 60))
    with pytest.warns(UserWarning, match="no feature/attribute association could be measured"):
        fs = FeatureSuppressor(protected_attributes=["group"], correlation_threshold=0.3)
        fs.fit_transform(single)

    assert fs.fit_result.correlation_before == {}
    assert fs.fit_result.correlation_after == {}
    assert fs.fit_result.correlation_reduction == {}
    assert any("only 1 observed level" in m for m in fs.fit_result.warnings)

    # A CONSTANT NUMERIC attribute is the same refusal, not a 0.0.
    const_num = _suppressor_frame(np.full(60, 7.0))
    with pytest.warns(UserWarning, match="no feature/attribute association could be measured"):
        fn = FeatureSuppressor(protected_attributes=["group"], correlation_threshold=0.3)
        fn.fit_transform(const_num)
    assert fn.fit_result.correlation_before == {}
    assert any("constant numeric value" in m for m in fn.fit_result.warnings)

    # CONTROL: two levels still measure the real association and still reduce it.
    two = _suppressor_frame(np.array(["a", "b"] * 30))
    fh = FeatureSuppressor(protected_attributes=["group"], correlation_threshold=0.3)
    fh.fit_transform(two)
    assert fh.fit_result.correlation_before["group"] == pytest.approx(0.5329, abs=1e-3)
    assert fh.fit_result.warnings == []


def test_feature_suppressor_does_not_certify_a_clean_feature_set_it_never_scored():
    """F7: the suppression DECISION. max_corr started at 0.0 and stayed 0.0, so
    no column ever cleared the threshold and the full feature set came back
    unchanged, indistinguishable from a genuine clean result."""
    single = _suppressor_frame(np.array(["a"] * 60))
    with pytest.warns(UserWarning):
        fs = FeatureSuppressor(protected_attributes=["group"], correlation_threshold=0.3)
        out = fs.fit_transform(single)

    assert list(out.columns) == ["proxy", "noise"]  # rows still pass through
    assert fs.fit_result.fit_metrics["features_not_assessed"] == ["proxy", "noise"]
    assert fs.fit_result.fit_metrics["n_features_not_assessed"] == 2
    assert fs.fit_result.fit_metrics["attributes_not_measured"] == ["group"]
    assert any("suppression not assessed" in m for m in fs.fit_result.warnings)
    assert any("neither suppressed nor certified" in m for m in fs.fit_result.warnings)

    # CONTROL: the identical feature values with a real two-level attribute
    # still identify and remove the proxy, and assess every column.
    two = _suppressor_frame(np.array(["a", "b"] * 30))
    fh = FeatureSuppressor(protected_attributes=["group"], correlation_threshold=0.3)
    outh = fh.fit_transform(two)
    assert list(outh.columns) == ["noise"]
    assert fh.fit_result.features_removed == ["proxy"]
    assert fh.fit_result.fit_metrics["features_not_assessed"] == []


def test_feature_suppressor_serialised_report_carries_no_structural_zero():
    """F8: to_dict() published correlation_before/after {'group': 0.0} and a
    0.0/0.0 reduction as measurements, with warnings == []."""
    df = pd.DataFrame(
        {
            "proxy": np.arange(60.0),
            "noise": np.random.default_rng(1).random(60),
            "group": ["a"] * 60,
        }
    )
    with pytest.warns(UserWarning):
        fs = FeatureSuppressor(protected_attributes=["group"])
        fs.fit_transform(df)

    d = fs.fit_result.to_dict()
    assert d["correlation_before"] == {} and d["correlation_after"] == {}
    assert 0.0 not in d["correlation_before"].values()
    assert d["warnings"], "the serialised report must carry the refusal, not just the object"
    assert d["fit_metrics"]["attributes_measured"] == []


# ---------------------------------------------------------------------------
# disparate_impact_removal
# ---------------------------------------------------------------------------


def test_disparate_impact_remover_names_a_group_too_small_to_have_a_distribution():
    """F9: one group-b row at f0=5.0 dragged all 59 group-a rows from mean
    -0.2294 to +2.4010. np.quantile of a single value returns that value at all
    101 knots, so the shared target rested on a fabricated flat curve. The only
    guard was vals.size == 0, which a 1-row group passes."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"f0": np.concatenate([rng.normal(0, 1, 59), [5.0]]), "group": ["a"] * 59 + ["b"]}
    )
    with pytest.warns(UserWarning, match="unreliable quantile estimate"):
        d = DisparateImpactRemover(protected_attributes=["group"])
        out = d.fit_transform(df)

    m = d.fit_result.fit_metrics
    assert m["n_per_group"]["f0"] == {"a": 59, "b": 1}
    assert m["features_repaired_on_unreliable_estimate"] == ["f0"]
    assert any("min_group_size=10" in w for w in d.fit_result.warnings)
    # The drag is real and is exactly what the warning is about.
    assert out[df["group"] == "a"]["f0"].mean() > 2.0

    # CONTROL: two well-populated groups repair cleanly and say nothing.
    dfh = pd.DataFrame(
        {
            "f0": np.concatenate([rng.normal(0, 1, 30), rng.normal(4, 1, 30)]),
            "group": ["a"] * 30 + ["b"] * 30,
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dh = DisparateImpactRemover(protected_attributes=["group"])
        outh = dh.fit_transform(dfh)
    assert dh.fit_result.fit_metrics["features_repaired_on_unreliable_estimate"] == []
    assert dh.fit_result.warnings == []
    assert _messages(w) == []
    # And the repair still works: the two group means land together.
    assert outh[dfh["group"] == "a"]["f0"].mean() == pytest.approx(
        outh[dfh["group"] == "b"]["f0"].mean(), abs=1e-6
    )


def test_disparate_impact_remover_transform_declares_rows_it_could_not_repair():
    """F16: fit on {a, b}, transform of ['a','b','c','c'] returned the two
    group-c rows byte-identical to their raw input, in the same column as the
    repaired ones, with no warning and no transform-time result at all."""
    fit = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "x": [1.0, 2, 3, 4, 5, 16, 17, 18, 19, 20]})
    new = pd.DataFrame({"g": ["a", "b", "c", "c"], "x": [1.0, 20.0, 3.0, 4.0]})
    r = DisparateImpactRemover(protected_attributes=["g"], min_group_size=2).fit(fit)

    with pytest.warns(UserWarning, match="never seen at fit"):
        out = r.transform(new)

    assert r.transform_result["n_rows_unrepaired"] == 2
    assert r.transform_result["unseen_groups"] == ["c"]
    assert r.transform_result["unrepaired_row_positions"] == [2, 3]
    # The fixture really does leave them raw: that is the disclosed state.
    assert list(out["x"])[2:] == [3.0, 4.0]

    # CONTROL: every group seen at fit -> no warning, nothing unrepaired, and
    # the repaired values are still produced.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seen = r.transform(pd.DataFrame({"g": ["a", "b"], "x": [1.0, 20.0]}))
    assert r.transform_result["n_rows_unrepaired"] == 0
    assert r.transform_result["unseen_groups"] == []
    assert _messages(w) == []
    assert list(seen["x"]) != [1.0, 20.0]


def test_disparate_impact_remover_refuses_a_min_group_size_of_one():
    """A minimum of 1 would re-open the defect by configuration."""
    with pytest.raises(ValueError, match="min_group_size must be at least 2"):
        DisparateImpactRemover(protected_attributes=["g"], min_group_size=1)


# ---------------------------------------------------------------------------
# label_massaging
# ---------------------------------------------------------------------------


def _massager_frame(n=60, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    if nan:
        X = pd.DataFrame(np.full((n, 3), np.nan), columns=["f1", "f2", "f3"])
    else:
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    X["grp"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    y = np.concatenate([rng.integers(0, 2, n // 2), (rng.random(n // 2) < 0.25).astype(int)])
    return X, y


def test_label_massager_refuses_a_ranker_with_no_decision_boundary():
    """F10: on all-NaN features the ranker trained, produced exactly 1 distinct
    score across 60 rows, and recorded ranker='logistic' with warnings == [] -
    a record byte-identical to the real-features run."""
    Xn, yn = _massager_frame(nan=True)
    with pytest.warns(UserWarning, match="single distinct score"):
        lm = LabelMassager(protected_attributes=["grp"]).fit(Xn, yn)

    # The fixture reaches the branch: the columns are PRESENT and empty, which
    # is exactly what the old column-count guard let through.
    assert list(Xn.columns[:3]) == ["f1", "f2", "f3"]
    assert Xn[["f1", "f2", "f3"]].isna().all().all()

    assert lm.fit_result.fit_metrics["ranker"] is None
    assert lm.fit_result.fit_metrics["n_labels_flipped"] == 0
    assert any("no borderline ordering exists" in w for w in lm.fit_result.warnings)
    assert np.array_equal(lm.get_massaged_labels(Xn, yn), yn)

    # CONTROL: real features still train a ranker and still relabel, and the two
    # records are no longer identical.
    Xr, yr = _massager_frame(nan=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lr = LabelMassager(protected_attributes=["grp"]).fit(Xr, yr)
        y_new = lr.get_massaged_labels(Xr, yr)
    assert lr.fit_result.fit_metrics["ranker"] == "logistic"
    assert int((y_new != yr).sum()) > 0
    assert lr.fit_result.warnings == []
    assert _messages(w) == []
    assert lr.fit_result.fit_metrics != lm.fit_result.fit_metrics


def test_label_massager_reports_the_labels_it_actually_rewrote():
    """F12: fit_metrics['n_labels_flipped'] was the PLANNED per-side count M
    while get_massaged_labels changed 2M labels. Reported 21 / actual 42, 19/38,
    12/24, 22/44 - always half, always in the reassuring direction."""
    for seed in (0, 1, 2, 7):
        rng = np.random.default_rng(seed)
        n = 300
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
        g = rng.integers(0, 2, n).astype(str)
        X["grp"] = g
        y = (rng.random(n) < np.where(g == "1", 0.6, 0.3)).astype(int)

        lm = LabelMassager(protected_attributes=["grp"]).fit(X, y)
        y_new = lm.get_massaged_labels(X, y)

        reported = lm.fit_result.fit_metrics["n_labels_flipped"]
        actual = int((y_new != y).sum())
        assert reported == actual, f"seed {seed}: reported {reported}, actual {actual}"
        assert actual > 0, "the fixture must actually flip labels"
        # Both sides are still visible separately, and the old per-side plan is
        # kept under a name that says what it is.
        assert (
            lm.fit_result.fit_metrics["n_promoted"] + lm.fit_result.fit_metrics["n_demoted"]
            == actual
        )
        assert lm.fit_result.fit_metrics["planned_flips_per_side"] * 2 >= actual
        assert lm.n_labels_flipped_ == actual


def test_label_massager_selection_does_not_depend_on_row_order():
    """F11: with a constant score vector, 'the instances closest to the decision
    boundary' was argsort tie-breaking on row order. Shuffling the same rows
    re-picked only 7 of its own 16 records; a healthy fit picks all of them."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    g = rng.integers(0, 2, n).astype(str)
    X["grp"] = g
    X["_id"] = np.arange(n)
    y = (rng.random(n) < np.where(g == "1", 0.6, 0.3)).astype(int)

    lm_a = LabelMassager(protected_attributes=["grp"]).fit(X.drop(columns="_id"), y)
    a = lm_a.get_massaged_labels(X.drop(columns="_id"), y)

    perm = rng.permutation(n)
    Xp = X.iloc[perm].reset_index(drop=True)
    yp = y[perm]
    lm_b = LabelMassager(protected_attributes=["grp"]).fit(Xp.drop(columns="_id"), yp)
    b = lm_b.get_massaged_labels(Xp.drop(columns="_id"), yp)

    picked_a = set(X["_id"].values[a != y])
    picked_b = set(Xp["_id"].values[b != yp])
    assert picked_a, "the fixture must flip something for this to mean anything"
    assert picked_a == picked_b, "healthy massaging must be permutation invariant"

    # The degenerate half: a constant score vector flips nothing at all, so
    # there is no order-dependent selection left to be invariant about.
    Xn, yn = _massager_frame(nan=True)
    with pytest.warns(UserWarning):
        lm_n = LabelMassager(protected_attributes=["grp"]).fit(Xn, yn)
    assert np.array_equal(lm_n.get_massaged_labels(Xn, yn), yn)


def test_label_massager_refuses_a_degenerate_score_vector_at_call_time():
    """The SECOND line of defence, which needs its own fixture.

    Sabotaging the call-time guard while testing it through the all-NaN frame
    came back GREEN, because the fit-time guard has already set _model to None
    by then and _massage returns before the call-time check is reached. The
    reverted line was simply not the line that fixes THAT case. This fixture
    reaches it: the ranker is fitted on REAL features (so it exists and has a
    boundary) and is then CALLED on a frame whose ranker columns are constant,
    where argsort would again be tie-breaking on row order.
    """
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    g = rng.integers(0, 2, n).astype(str)
    X["grp"] = g
    y = (rng.random(n) < np.where(g == "1", 0.6, 0.3)).astype(int)

    lm = LabelMassager(protected_attributes=["grp"]).fit(X, y)
    # The fixture reaches the branch: a real trained ranker, and a call-time
    # frame that gives every row the same score.
    assert lm._model is not None  # noqa: SLF001
    flat = X.copy()
    flat[["f1", "f2", "f3"]] = 0.25
    scores = lm._model.predict_proba(flat[["f1", "f2", "f3"]].values)[:, 1]  # noqa: SLF001
    assert np.unique(scores).size == 1

    with pytest.warns(UserWarning, match="same score"):
        out = lm.get_massaged_labels(flat, y)
    assert np.array_equal(out, y), "no ordering exists, so nothing may be relabelled"
    assert lm.n_labels_flipped_ == 0

    # CONTROL: the same fitted massager on the real frame still relabels.
    y_new = lm.get_massaged_labels(X, y)
    assert int((y_new != y).sum()) > 0
    assert lm.n_labels_flipped_ == int((y_new != y).sum())


def test_label_massager_refuses_to_score_rows_whose_features_were_never_supplied():
    """F11 (second half): a column absent from X at call time was reindexed in
    as 0.0. With the dominant predictor removed, 14 labels were still rewritten
    and 26 record-level decisions changed, with no error and no warning."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    g = rng.integers(0, 2, n).astype(str)
    X["grp"] = g
    y = (rng.random(n) < np.where(g == "1", 0.6, 0.3)).astype(int)

    lm = LabelMassager(protected_attributes=["grp"]).fit(X, y)
    with pytest.raises(ValueError, match=r"absent from X.*f1"):
        lm.get_massaged_labels(X.drop(columns=["f1"]), y)

    # A NaN is filled as at fit time, but counted and named.
    holed = X.copy()
    holed.loc[0, "f1"] = np.nan
    with pytest.warns(UserWarning, match="missing ranker feature"):
        lm.get_massaged_labels(holed, y)
    assert lm.n_features_imputed_ == 1

    # CONTROL: the complete frame imputes nothing and still relabels.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_new = lm.get_massaged_labels(X, y)
    assert lm.n_features_imputed_ == 0
    assert _messages(w) == []
    assert int((y_new != y).sum()) > 0


# ---------------------------------------------------------------------------
# residual_transform
# ---------------------------------------------------------------------------


def test_residual_transformer_refuses_an_attribute_with_one_group():
    """F13: correlation_before/after/reduction all {'race': 0.0}, identical for
    a gaussian feature AND a monotone ramp, so the statistic could not vary with
    the data at all. warnings == []."""
    n = 200
    gauss = pd.DataFrame({"race": ["A"] * n, "income": np.random.default_rng(5).normal(75, 25, n)})
    ramp = pd.DataFrame({"race": ["A"] * n, "income": np.arange(n, dtype=float)})

    for df in (gauss, ramp):
        with pytest.warns(UserWarning, match="no feature/attribute association"):
            t = ResidualTransformer(protected_attributes=["race"]).fit(df)
        assert t.fit_result.correlation_before == {}
        assert t.fit_result.correlation_after == {}
        assert t.fit_result.correlation_reduction == {}
        assert any("only 1 observed level" in w for w in t.fit_result.warnings)

    # CONTROL: two real groups still measure the association and still remove it.
    rng = np.random.default_rng(5)
    dfh = pd.DataFrame({"race": ["A", "B"] * (n // 2), "income": rng.normal(75, 25, n)})
    dfh.loc[dfh["race"] == "B", "income"] += 25
    th = ResidualTransformer(protected_attributes=["race"]).fit(dfh)
    assert th.fit_result.correlation_before["race"] > 0.4
    assert th.fit_result.correlation_after["race"] < 1e-6
    assert th.fit_result.correlation_reduction["race"] > 0.99
    assert th.fit_result.warnings == []


# ---------------------------------------------------------------------------
# correlation_reduction
# ---------------------------------------------------------------------------


def test_correlation_reducer_issues_no_pass_verdict_it_never_compared():
    """F14: fit_metrics['target_met'] == {'race': True}, a PASS for a target
    nothing was ever compared against, on both the one-level categorical and
    the constant-numeric path, with warnings == []."""
    rng = np.random.default_rng(0)
    n = 300
    cat = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "race": ["A"] * n})
    num = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "race": np.full(n, 3.0)}
    )

    for df, why in ((cat, "only 1 observed level"), (num, "constant numeric value")):
        with pytest.warns(UserWarning, match="no feature/attribute association"):
            r = CorrelationReducer(protected_attributes=["race"], target_correlation=0.1).fit(df)
        res = r.fit_result
        assert res.fit_metrics["target_met"] == {"race": None}
        assert res.fit_metrics["attributes_not_measured"] == ["race"]
        assert res.correlation_before == {} and res.correlation_after == {}
        assert res.correlation_reduction == {}
        assert any(why in w for w in res.warnings)
        assert any("target_met is None" in w for w in res.warnings)

    # CONTROL: three real groups measure a real association, reduce it, and the
    # verdict is a genuine True with an empty warnings list.
    df3 = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "race": ["A", "B", "C"] * (n // 3)}
    )
    df3.loc[df3["race"] == "B", "f1"] += 2
    r3 = CorrelationReducer(protected_attributes=["race"], target_correlation=0.1).fit(df3)
    assert r3.fit_result.fit_metrics["target_met"] == {"race": True}
    assert r3.fit_result.correlation_before["race"] > 0.4
    assert r3.fit_result.correlation_after["race"] <= 0.1
    assert r3.fit_result.warnings == []


def test_a_constant_feature_still_reads_zero_against_a_real_attribute():
    """The half of the split that must NOT change. A constant FEATURE genuinely
    has no association with anything and must keep measuring 0.0; only a
    constant ATTRIBUTE is a could-not-check. Collapsing both into a refusal
    would be the same defect running backwards."""
    n = 200
    df = pd.DataFrame(
        {
            "flat": np.full(n, 13.279640262203964),  # the ulp-noise value from audit-6
            "real": np.random.default_rng(2).normal(size=n),
            "race": ["A", "B"] * (n // 2),
        }
    )
    r = CorrelationReducer(protected_attributes=["race"], target_correlation=0.9).fit(df)
    # 'flat' contributes a measured 0.0, so the mean over both features is
    # strictly between 0 and the 'real' feature's own association.
    assert "race" in r.fit_result.correlation_before
    assert r.fit_result.correlation_before["race"] >= 0.0
    assert r.fit_result.fit_metrics["target_met"]["race"] is not None

    dfn = pd.DataFrame({"flat": np.full(n, 3.0), "race": ["A", "B"] * (n // 2)})
    rn = CorrelationReducer(protected_attributes=["race"], target_correlation=0.9).fit(dfn)
    assert rn.fit_result.correlation_before == {"race": 0.0}


# ---------------------------------------------------------------------------
# intersectional_transform
# ---------------------------------------------------------------------------


def _intersectional_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    parts = [
        pd.DataFrame({"gender": g, "race": a, "score": rng.normal(mu, 2.0, 60)})
        for g, a, mu in [("F", "A", 9.6), ("F", "B", 14.8), ("M", "A", -0.5), ("M", "B", 4.9)]
    ]
    parts.append(pd.DataFrame({"gender": ["X"], "race": ["A"], "score": [999.0]}))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("strategy", ["merge", "exclude", "keep"])
def test_intersectional_transformer_counts_the_rows_it_left_untransformed(strategy):
    """F17: a 1-row intersection has std NaN (ddof=1) and `nan > 0` is False, so
    999.0 went in and 999.0 came out while every other group mean was rescaled
    to 11.131 - one column carrying two scales, with features_modified still
    naming it and warnings == []. min_group_size=1 proves this is not the
    small-group policy: the bare `> 0` test is the swallower."""
    df = _intersectional_frame()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = IntersectionalTransformer(
            protected_attributes=["gender", "race"], handle_small_groups=strategy
        )
        out = t.fit_transform(df)

    # The fixture reaches the branch under test. Two shapes, both of which used
    # to fall through silently:
    #   - merge/keep: the 1-row group HAS stats, with a NaN std, and `nan > 0`
    #     answered False without distinguishing NaN from a real zero spread;
    #   - exclude: the group has no stats entry at all, so the lookup missed.
    stats = t._group_stats["score"]  # noqa: SLF001 - asserting the fixture hits the branch
    if strategy == "exclude":
        assert t._intersectional_groups[("X", "A")] is None  # noqa: SLF001
        assert all(s["count"] >= 2 for s in stats.values())
    else:
        lone = [s for s in stats.values() if s["count"] == 1]
        assert lone, "the 1-row intersection must reach the stats table"
        assert np.isnan(lone[0]["std"]) and lone[0]["transformable"] is False

    assert df["score"].iloc[-1] == 999.0 and out["score"].iloc[-1] == 999.0
    assert t.rows_not_transformed_.tolist() == [240]
    assert t.groups_not_transformed_ == ["X_A"]
    assert t.fit_result.fit_metrics["n_rows_not_transformed"] == 1
    assert t.fit_result.warnings, "the miss must be on the result, not only in a warning"
    assert any("UNTRANSFORMED" in m for m in _messages(w))
    if strategy == "exclude":
        # The sharpest case: the metrics claimed the group was excluded while
        # its row was physically present and raw.
        assert t.fit_result.fit_metrics["groups_excluded_but_returned"] == ["X_A"]
        assert any("RETURNED UNTRANSFORMED rather than removed" in w for w in t.fit_result.warnings)


def test_intersectional_transformer_min_group_size_one_is_still_disclosed():
    """With min_group_size=1 the group is not 'small' by policy at all, so any
    fix that only touched the small-group machinery would be green here."""
    df = _intersectional_frame()
    with pytest.warns(UserWarning, match="UNTRANSFORMED"):
        t = IntersectionalTransformer(protected_attributes=["gender", "race"], min_group_size=1)
        out = t.fit_transform(df)
    assert out["score"].iloc[-1] == 999.0
    assert t.fit_result.fit_metrics["n_rows_not_transformed"] == 1


def test_intersectional_transformer_healthy_groups_are_all_normalised():
    """CONTROL: four groups of 60 all land on one mean, nothing is skipped, and
    nothing is warned about."""
    rng = np.random.default_rng(2)
    parts = [
        pd.DataFrame({"gender": g, "race": a, "score": rng.normal(mu, 2.0, 60)})
        for g, a, mu in [("F", "A", 9.6), ("F", "B", 14.8), ("M", "A", -0.5), ("M", "B", 4.9)]
    ]
    df = pd.concat(parts, ignore_index=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = IntersectionalTransformer(protected_attributes=["gender", "race"])
        out = t.fit_transform(df)

    means = out.assign(g=df["gender"] + df["race"]).groupby("g")["score"].mean()
    assert means.max() - means.min() < 1e-6
    assert t.rows_not_transformed_.size == 0
    assert t.groups_not_transformed_ == []
    assert t.fit_result.warnings == []
    assert _messages(w) == []
