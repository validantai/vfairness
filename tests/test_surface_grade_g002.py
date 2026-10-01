"""Surface grade g002: feature-engineering transformers.

Batch g002 covers every public fit/transform/getter on
``vfairness.preprocessing.feature_engineering.transformers``. The defect class
being pinned is the one the whole campaign exists for: a value nobody measured,
replaced by a neutral default, then reported as if it were a measurement.

Three fabrications were found by execution on 2026-09-17 and are fixed here:

  g002-1  ``CorrelationReducer.fit`` published ``features_modified`` for every
          feature column unconditionally, including columns it could build NO
          reduction for (fewer than 10 valid rows reach ``np.linalg.lstsq``).
          ``transform()`` returned those byte-identical while the report named
          them as modified and ``target_met`` read True.
  g002-2  ``CorrelationReducer._transform_decorrelate`` selected the projection
          with ``hasattr(self, "_ZtZ_inv")``, which stays True forever once any
          fit succeeded, so a REFIT on a frame too small to estimate a
          projection silently kept applying the PREVIOUS fit's projection.
  g002-3  ``ReweightingTransformer.fit`` and ``IntersectionalTransformer.fit``
          were the only two transformers in the file that did not disclose
          their own fewer-than-2-groups case. One observed group makes every
          inverse-frequency weight exactly 1.0 (the "no reweighting needed"
          value) and makes intersectional normalisation a plain rescaling, and
          both came back with ``fit_result.warnings == []``.

Every refusal pin is paired with a healthy-data CONTROL that asserts a real
value is still measured, computed independently of the code under test.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.transformers import (
    CorrelationReducer,
    DisparateImpactRemover,
    FairRepresentationTransformer,
    FeatureImportanceResult,
    FeatureSuppressor,
    IntersectionalTransformer,
    LabelMassager,
    Resampler,
    ResidualTransformer,
    ReweightingTransformer,
    TransformationResult,
)

# ---------------------------------------------------------------------------
# fixtures + an independent judge
# ---------------------------------------------------------------------------


def _healthy(n: int = 200, seed: int = 0):
    """A frame carrying a real, findable disparity: 'proxy' separates gender."""
    rng = np.random.default_rng(seed)
    gender = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    df = pd.DataFrame(
        {
            "proxy": np.where(gender == "a", 0.0, 6.0) + rng.normal(0, 1.0, n),
            "clean": rng.normal(0, 1.0, n),
            "gender": gender,
            "age": rng.normal(40, 10, n),
        }
    )
    y = (rng.random(n) < np.where(gender == "a", 0.25, 0.75)).astype(int)
    return df, y


def _eta(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio eta, written out here so the control does not copy the
    number the code under test produces."""
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(groups)[ok]
    grand = x.mean()
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 0.0:
        return 0.0
    ss_between = sum(len(x[g == lvl]) * (x[g == lvl].mean() - grand) ** 2 for lvl in set(g))
    return float(np.sqrt(min(max(ss_between / ss_total, 0.0), 1.0)))


def _messages(caught) -> list:
    return [str(w.message) for w in caught]


# ===========================================================================
# g002-1  CorrelationReducer: a feature it could not reduce is not "modified"
# ===========================================================================


def _frame_with_an_unreducible_column():
    """'partial' carries 9 observed rows, one below the 10 that
    ``_fit_residualize`` needs, so no beta is fitted for it and transform()
    hands it back untouched. The other three columns are fully reducible."""
    df, _ = _healthy()
    partial = np.full(len(df), np.nan)
    partial[:9] = np.where(df["gender"].to_numpy()[:9] == "a", 0.0, 50.0) + np.arange(9)
    df["partial"] = partial
    return df


def test_a_feature_no_reduction_was_fitted_for_is_not_reported_as_modified():
    """g002-1. Measured before the fix: features_modified ==
    ['proxy', 'clean', 'age', 'partial'], target_met == {'gender': True}, and
    the sole warning was about the ASSOCIATION being unscorable, not about the
    reduction never happening, while transform() returned 'partial' byte for
    byte."""
    df = _frame_with_an_unreducible_column()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = CorrelationReducer(protected_attributes=["gender"], method="residualize").fit(df)

    # The fixture really does reach the branch under test.
    assert r._residual_coefficients["partial"] is None  # noqa: SLF001
    out = r.transform(df)
    raw = df["partial"].to_numpy()
    assert np.array_equal(raw[:9], out["partial"].to_numpy()[:9]), (
        "the fixture no longer reaches the untouched-column branch"
    )

    res = r.fit_result
    assert "partial" not in res.features_modified, (
        f"a column transform() returned unchanged is reported as modified: {res.features_modified}"
    )
    assert res.fit_metrics["features_not_reduced"] == ["partial"]
    assert res.fit_metrics["n_features_not_reduced"] == 1
    assert any("no reduction was fitted" in w for w in res.warnings), (
        "the refusal never reached the serialised result"
    )
    assert any("no reduction was fitted" in m for m in _messages(caught)), (
        "the refusal never reached a Python warning"
    )
    # The three real columns are still reported as reduced. A guard that
    # emptied features_modified would pass the assertions above and destroy
    # the finding the report carries.
    assert res.features_modified == ["proxy", "clean", "age"]


def test_control_a_healthy_correlation_reduction_still_measures_and_reduces():
    """CONTROL for g002-1: every column reducible, a real association measured
    against an independently computed eta, and nothing warned about."""
    df, _ = _healthy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = CorrelationReducer(protected_attributes=["gender"], method="residualize").fit(df)

    res = r.fit_result
    expected_before = float(
        np.mean(
            [_eta(df[c].to_numpy(), df["gender"].to_numpy()) for c in ("proxy", "clean", "age")]
        )
    )
    assert res.correlation_before["gender"] == pytest.approx(expected_before, abs=1e-12)
    assert expected_before > 0.3, "the fixture must carry a findable disparity"
    assert res.correlation_after["gender"] < 1e-9
    assert res.correlation_reduction["gender"] > 0.99
    assert res.features_modified == ["proxy", "clean", "age"]
    assert res.fit_metrics["features_not_reduced"] == []
    assert res.fit_metrics["target_met"] == {"gender": True}
    assert res.warnings == [], f"the healthy run warned: {res.warnings}"
    assert _messages(caught) == []


# ===========================================================================
# g002-2  CorrelationReducer: a refit does not inherit the old projection
# ===========================================================================


def test_a_refit_that_cannot_estimate_a_projection_does_not_reuse_the_old_one():
    """g002-2. ``hasattr(self, "_ZtZ_inv")`` is True forever after the first
    successful fit. Measured before the fix: fit on 200 rows, refit on 8, and
    transform() still applied the 200-row projection (output != raw) with
    nothing on fit_result saying the second fit estimated nothing."""
    df, _ = _healthy()
    r = CorrelationReducer(protected_attributes=["gender"], method="decorrelate").fit(df)
    assert r._ZtZ_inv is not None  # noqa: SLF001 - the first fit really succeeded

    small = df.head(8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r.fit(small)

    assert r._ZtZ_inv is None, (  # noqa: SLF001
        "the refit kept the previous fit's projection matrix"
    )
    out = r.transform(small)
    raw = small[["proxy", "clean", "age"]].to_numpy(dtype=float)
    assert np.array_equal(out.to_numpy(dtype=float), raw), (
        "a projection estimated from data this fit never saw was applied anyway"
    )
    assert r.fit_result.features_modified == []
    assert any("no reduction was fitted" in w for w in r.fit_result.warnings)
    assert any("no reduction was fitted" in m for m in _messages(caught))


def test_control_decorrelate_on_a_healthy_frame_still_projects():
    """CONTROL for g002-2: the projection must still fire and still work."""
    df, _ = _healthy()
    r = CorrelationReducer(protected_attributes=["gender"], method="decorrelate").fit(df)
    out = r.transform(df)
    assert r._ZtZ_inv is not None  # noqa: SLF001
    assert not np.allclose(out["proxy"].to_numpy(), df["proxy"].to_numpy())
    assert _eta(out["proxy"].to_numpy(), df["gender"].to_numpy()) < 1e-6
    assert r.fit_result.features_modified == ["proxy", "clean", "age"]
    assert r.fit_result.warnings == []


# ===========================================================================
# g002-3a  ReweightingTransformer: 1.0 for a comparison that never happened
# ===========================================================================


def test_one_protected_group_does_not_earn_a_no_reweighting_needed_weight():
    """g002-3. Measured before the fix: group_weights {'a': 1.0}, every sample
    weight 1.0, fit_result.warnings == [] and no Python warning, byte-identical
    to a frame that genuinely needed no correction."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(protected_attributes=["gender"], method="inverse_frequency").fit(
            df
        )

    assert t.fit_result.fit_metrics["n_groups_observed"] == 1
    assert t.fit_result.fit_metrics["between_group_balance_measured"] is False
    assert any("1 observed group" in w for w in t.fit_result.warnings), (
        f"nothing on the result says the comparison never happened: {t.fit_result.warnings}"
    )
    assert any("could-not-check" in w for w in t.fit_result.warnings)
    assert any("1 observed group" in m for m in _messages(caught))

    # The weight itself is still the correct value of the formula. Refusing a
    # number that IS computable would throw evidence away; the missing piece
    # was the third state, and it must reach the ndarray surface too.
    with warnings.catch_warnings(record=True) as caught_w:
        warnings.simplefilter("always")
        w = t.get_sample_weights(df)
    assert np.allclose(w, 1.0)
    assert any("1.0 by construction" in m for m in _messages(caught_w)), (
        "an all-1.0 weight array reached the caller with no signal at all"
    )


def test_control_two_unbalanced_groups_still_get_their_exact_weights():
    """CONTROL for g002-3: 100 'a' rows and 40 'b' rows, so the inverse
    frequency weights are n / (n_groups * n_g) = 140/200 and 140/80 exactly."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"f": rng.normal(size=140), "gender": ["a"] * 100 + ["b"] * 40})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(protected_attributes=["gender"], method="inverse_frequency").fit(
            df
        )
        w = t.get_sample_weights(df)

    assert t.fit_result.fit_metrics["between_group_balance_measured"] is True
    assert t.fit_result.fit_metrics["n_groups_observed"] == 2
    assert w[:100] == pytest.approx(0.7, abs=1e-12)
    assert w[100:] == pytest.approx(1.75, abs=1e-12)
    assert w.sum() == pytest.approx(140.0, abs=1e-9)
    assert t.fit_result.warnings == []
    assert _messages(caught) == []


def test_target_parity_on_a_single_label_says_nothing_was_equalized():
    """A single observed class makes P(y=c) == 1, so every Kamiran & Calders
    weight is 1.0 by construction. Before the fix that reached the caller as an
    all-1.0 array with warnings == []."""
    df, _ = _healthy(n=100)
    y_const = np.zeros(len(df), dtype=int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(protected_attributes=["gender"], method="target_parity").fit(
            df, y_const
        )

    assert t.fit_result.fit_metrics["n_labels_observed"] == 1
    assert any("1 observed label" in w for w in t.fit_result.warnings)
    assert any("could-not-check" in w for w in t.fit_result.warnings)
    assert any("1 observed label" in m for m in _messages(caught))
    assert np.allclose(list(t._group_weights.values()), 1.0)  # noqa: SLF001


def test_control_target_parity_on_two_labels_still_equalizes_the_rates():
    """CONTROL: the Kamiran & Calders weights must still make the weighted
    per-group positive rates equal, checked by recomputing them here."""
    df, y = _healthy(n=200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(protected_attributes=["gender"], method="target_parity").fit(
            df, y
        )
        w = t.get_sample_weights(df, y)

    rates = []
    for g in ("a", "b"):
        m = df["gender"].to_numpy() == g
        rates.append(float((w[m] * y[m]).sum() / w[m].sum()))
    assert rates[0] == pytest.approx(rates[1], abs=1e-9)
    # ... and the raw rates really were far apart, so the fixture had work to do.
    raw = [float(y[df["gender"].to_numpy() == g].mean()) for g in ("a", "b")]
    assert abs(raw[0] - raw[1]) > 0.3
    assert t.fit_result.fit_metrics["n_labels_observed"] == 2
    assert t.fit_result.warnings == []
    assert _messages(caught) == []


# ===========================================================================
# g002-3b  IntersectionalTransformer: one subgroup is not an intersection
# ===========================================================================


def test_a_single_intersectional_group_says_no_adjustment_was_made():
    """g002-3. With one (gender, race) cell the within-group standardisation is
    a whole-sample rescaling: nothing is brought onto a common scale because
    there is no other subgroup. Measured before the fix: every row rescaled,
    features_modified naming the column, fit_result.warnings == [] and no
    Python warning."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"score": rng.normal(0, 2.0, 50), "gender": ["m"] * 50, "race": ["x"] * 50})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = IntersectionalTransformer(
            protected_attributes=["gender", "race"], min_group_size=30
        ).fit(df)

    assert t.fit_result.fit_metrics["n_intersectional_groups"] == 1
    assert t.fit_result.fit_metrics["intersectional_contrast_measured"] is False
    assert any("1 usable intersectional group" in w for w in t.fit_result.warnings), (
        f"nothing on the result says no adjustment was possible: {t.fit_result.warnings}"
    )
    assert any("could-not-check" in w for w in t.fit_result.warnings)
    assert any("NO intersectional adjustment was made" in m for m in _messages(caught))


def test_control_four_intersectional_groups_are_still_levelled_and_silent():
    """CONTROL for g002-3b: four real subgroups with different centres must
    still land on one common mean, and nothing may warn."""
    rng = np.random.default_rng(4)
    parts = [
        pd.DataFrame({"gender": g, "race": a, "score": rng.normal(mu, 2.0, 50)})
        for g, a, mu in [("f", "x", 9.6), ("f", "y", 14.8), ("m", "x", -0.5), ("m", "y", 4.9)]
    ]
    df = pd.concat(parts, ignore_index=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = IntersectionalTransformer(protected_attributes=["gender", "race"], min_group_size=30)
        out = t.fit_transform(df)

    means = out.assign(k=df["gender"] + df["race"]).groupby("k")["score"].mean()
    assert float(means.max() - means.min()) < 1e-9
    assert t.fit_result.fit_metrics["intersectional_contrast_measured"] is True
    assert t.fit_result.fit_metrics["n_intersectional_groups"] == 4
    assert t.rows_not_transformed_.size == 0
    assert t.fit_result.warnings == []
    assert _messages(caught) == []


# ===========================================================================
# Refusals that were already honest, pinned so a refactor cannot undo them
# ===========================================================================


def test_residual_transformer_leaves_an_unmeasurable_attribute_out_of_the_report():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ResidualTransformer(protected_attributes=["gender"]).fit(df)
    res = t.fit_result
    assert res.correlation_before == {} and res.correlation_after == {}
    assert res.correlation_reduction == {}, "a 0.0 reduction was published for no comparison"
    assert any("only 1 observed level" in w for w in res.warnings)
    assert any("rather than reported as 0.0" in m for m in _messages(caught))


def test_control_residual_transformer_still_removes_a_real_group_gap():
    df, _ = _healthy()
    t = ResidualTransformer(protected_attributes=["gender"]).fit(df)
    out = t.transform(df)
    before = _eta(df["proxy"].to_numpy(), df["gender"].to_numpy())
    after = _eta(out["proxy"].to_numpy(), df["gender"].to_numpy())
    assert before > 0.9 and after < 1e-9
    assert t.fit_result.warnings == []


def test_feature_suppressor_does_not_certify_a_column_it_could_not_score():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})
    t = FeatureSuppressor(protected_attributes=["gender"], strategy="remove").fit(df)
    assert t.fit_result.fit_metrics["features_not_assessed"] == ["f"]
    assert t.fit_result.features_removed == []
    assert any("neither suppressed nor certified" in w for w in t.fit_result.warnings)


def test_control_feature_suppressor_still_removes_a_real_proxy():
    df, _ = _healthy()
    t = FeatureSuppressor(
        protected_attributes=["gender"], strategy="remove", correlation_threshold=0.3
    ).fit(df)
    assert t.fit_result.features_removed == ["proxy"]
    assert t.fit_result.fit_metrics["features_not_assessed"] == []
    assert list(t.transform(df).columns) == ["clean", "age"]
    assert t.fit_result.warnings == []


def test_disparate_impact_remover_calls_a_one_group_frame_a_could_not_check():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})
    t = DisparateImpactRemover(protected_attributes=["gender"]).fit(df)
    assert t.fit_result.fit_metrics["n_groups"] == 1
    assert t.fit_result.fit_metrics["n_features_repaired"] == 0
    assert any("fewer than 2 groups" in w for w in t.fit_result.warnings), (
        f"the fit published no refusal at all: {t.fit_result.warnings}"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = t.transform(df)
    assert np.allclose(out["f"].to_numpy(), df["f"].to_numpy())
    assert t.transform_result["n_rows_unrepaired"] == 60, (
        "a transform that repaired nothing reported 0 rows unrepaired"
    )
    assert "could-not-check" in (t.transform_result["reason"] or "")
    assert any("NO row was repaired" in m for m in _messages(caught))


def test_control_disparate_impact_remover_still_equalizes_two_group_means():
    df, _ = _healthy()
    t = DisparateImpactRemover(protected_attributes=["gender"], repair_level=1.0).fit(df)
    out = t.transform(df).assign(gender=df["gender"].to_numpy())
    raw = df.groupby("gender")["proxy"].mean()
    rep = out.groupby("gender")["proxy"].mean()
    assert abs(raw["a"] - raw["b"]) > 5.0
    assert abs(rep["a"] - rep["b"]) < 0.1
    assert t.transform_result["n_rows_unrepaired"] == 0
    assert t.fit_result.warnings == []


def test_label_massager_reports_the_flips_it_made_not_half_of_them():
    df, y = _healthy()
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)
    y_new = lm.get_massaged_labels(df, y)
    actual = int((y_new != y).sum())
    assert actual > 0
    assert lm.n_labels_flipped_ == actual
    assert lm.fit_result.fit_metrics["n_labels_flipped"] == actual
    assert lm.fit_result.fit_metrics["planned_flips_per_side"] * 2 == actual
    before = [float(y[df["gender"].to_numpy() == g].mean()) for g in ("a", "b")]
    after = [float(y_new[df["gender"].to_numpy() == g].mean()) for g in ("a", "b")]
    assert abs(before[0] - before[1]) > 0.3
    assert abs(after[0] - after[1]) < 0.05


def test_label_massager_without_a_usable_ranker_says_so_instead_of_staying_silent():
    df, y = _healthy()
    blind = df.copy()
    for c in ("proxy", "clean", "age"):
        blind[c] = np.nan
    lm = LabelMassager(protected_attributes=["gender"]).fit(blind, y)
    assert lm.fit_result.fit_metrics["ranker"] is None
    assert lm.fit_result.fit_metrics["n_labels_flipped"] == 0
    assert any("no borderline ordering exists" in w for w in lm.fit_result.warnings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_new = lm.get_massaged_labels(blind, y)
    assert np.array_equal(y_new, y)
    assert any("could-not-check" in m for m in _messages(caught))


def test_resampler_says_between_group_balance_was_not_measured_on_one_group():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})
    y = (rng.random(60) < 0.5).astype(int)
    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rs.get_resampled_data(df, y)
    assert any("between-group balance was NOT measured" in m for m in _messages(caught))
    # One key, because there is only one group: the cells were levelled against
    # each other, which says nothing at all about representation across groups.
    assert list(rs.resample_result["group_totals_after"]) == ["a"]
    assert any("between-group balance was NOT measured" in w for w in rs.fit_result.warnings), (
        "the refusal reached a Python warning only, not the serialised result"
    )


def test_control_resampler_still_balances_two_groups_and_counts_the_copies():
    df, y = _healthy()
    keep = np.r_[
        np.flatnonzero(df["gender"].to_numpy() == "a"),
        np.flatnonzero(df["gender"].to_numpy() == "b")[:40],
    ]
    dfu, yu = df.iloc[keep].reset_index(drop=True), y[keep]
    rs = Resampler(protected_attributes=["gender"]).fit(dfu, yu)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        x_bal, y_bal = rs.get_resampled_data(dfu, yu)
    sizes = set(rs.resample_result["cell_sizes_after"].values())
    assert len(sizes) == 1, f"cells were not balanced: {rs.resample_result['cell_sizes_after']}"
    assert len(x_bal) == len(y_bal) == sum(rs.resample_result["cell_sizes_after"].values())
    assert rs.resample_result["n_duplicated_rows"] > 0
    assert rs.resample_result["n_effective_rows"] == len(dfu)


def test_fair_representation_refuses_what_it_cannot_encode():
    df, _ = _healthy(n=80)
    with pytest.raises(ValueError, match="at least one row"):
        FairRepresentationTransformer(protected_attributes=["gender"], epochs=2).fit(df.iloc[:0])

    f = FairRepresentationTransformer(
        protected_attributes=["gender"], representation_dim=2, epochs=5
    ).fit(df)
    with pytest.raises(ValueError, match="absent"):
        f.transform(df.drop(columns=["proxy"]))

    rng = np.random.default_rng(9)
    one = pd.DataFrame({"f": rng.normal(size=40), "gender": ["a"] * 40})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=5
        ).fit(one)
    assert g.fit_result.fit_metrics["n_groups"] == 1
    assert g.fit_result.fit_metrics["adversary_output_dim"] == 2
    assert any("NO fairness constraint was applied" in m for m in _messages(caught))


def test_control_fair_representation_still_encodes_and_counts_imputations():
    df, _ = _healthy(n=120)
    f = FairRepresentationTransformer(
        protected_attributes=["gender"], representation_dim=2, epochs=5
    ).fit(df)
    z = f.transform(df)
    assert list(z.columns) == ["rep_0", "rep_1"]
    assert np.isfinite(z.to_numpy()).all()
    # The latent code must carry the rows, not a constant: an all-zero or
    # otherwise degenerate representation is finite too, and would pass an
    # is-finite check while encoding nothing at all.
    assert np.unique(z.to_numpy()).size > len(df) // 2
    assert float(np.ptp(z["rep_0"].to_numpy())) > 0.1
    assert float(np.ptp(z["rep_1"].to_numpy())) > 0.1
    # ... and it must still reconstruct the input it was trained on: the row
    # ordering of the strongest feature has to survive into the latent space.
    corrs = [
        abs(float(np.corrcoef(z[c].to_numpy(), df["proxy"].to_numpy())[0, 1]))
        for c in ("rep_0", "rep_1")
    ]
    assert max(corrs) > 0.3, f"the representation lost the input entirely: {corrs}"
    assert f.transform_result["n_values_imputed"] == 0
    assert f.fit_result.warnings == []

    holed = df.copy()
    holed.loc[holed.index[:3], "proxy"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.transform(holed)
    assert f.transform_result["n_values_imputed"] == 3
    assert any("imputed with the fit-time mean" in m for m in _messages(caught))


# ===========================================================================
# The base class and the two converters
# ===========================================================================


def test_base_fit_transform_and_feature_names_out_carry_the_real_columns():
    df, _ = _healthy()
    r = CorrelationReducer(protected_attributes=["gender"])
    unfitted = CorrelationReducer(protected_attributes=["gender"])
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.get_feature_names_out()
    out = r.fit_transform(df)
    assert list(out.columns) == ["proxy", "clean", "age"]
    assert r.get_feature_names_out() == ["proxy", "clean", "age"]
    # fit_transform must be fit() then transform(), not a second independent path.
    assert np.allclose(out.to_numpy(dtype=float), r.transform(df).to_numpy(dtype=float))


def test_the_converters_do_not_turn_an_unmeasured_value_into_a_number():
    """Both to_dict() methods are field copies. The pin is that they stay that
    way: a NaN measurement and a None verdict must survive serialisation rather
    than being defaulted to 0.0 or False on the way out."""
    tr = TransformationResult(
        method="m",
        n_features_original=3,
        n_features_transformed=2,
        n_samples=10,
        correlation_before={"g": 0.5},
        correlation_after={"g": float("nan")},
        fit_metrics={"target_met": {"g": None}},
        warnings=["w"],
    )
    d = tr.to_dict()
    assert np.isnan(d["correlation_after"]["g"])
    assert d["fit_metrics"]["target_met"]["g"] is None
    assert d["correlation_before"]["g"] == 0.5
    # ... and the derived reduction excludes the unmeasured side rather than
    # calling it a 0.0 reduction.
    assert tr.correlation_reduction == {}

    fi = FeatureImportanceResult(
        feature="f",
        predictive_importance=float("nan"),
        fairness_impact=0.3,
        correlation_with_protected={"g": float("nan")},
        recommendation="keep",
        rationale="r",
    )
    fd = fi.to_dict()
    assert np.isnan(fd["predictive_importance"])
    assert np.isnan(fd["correlation_with_protected"]["g"])
    assert fd["fairness_impact"] == 0.3
    assert fd["recommendation"] == "keep"


def test_the_reweighting_and_massaging_transforms_are_identity_on_features():
    """Both classes expose their product through a getter, so transform() is
    documented as a pass-through. Pinned, because a silent change here would
    look like a transformation nobody asked for."""
    df, y = _healthy()
    rw = ReweightingTransformer(protected_attributes=["gender"]).fit(df)
    assert rw.transform(df).equals(df[["proxy", "clean", "age"]])
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)
    assert lm.transform(df).equals(df[["proxy", "clean", "age"]])
    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    assert rs.transform(df).equals(df[["proxy", "clean", "age"]])


def test_get_sample_weights_refuses_a_group_it_never_fitted():
    """A (group, label) cell with no fitted weight must come back NaN, never the
    1.0 that reads as "no reweighting needed". Pinned here because
    get_sample_weights is the surface a caller trains on."""
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"f": rng.normal(size=120), "gender": ["a"] * 80 + ["b"] * 40})
    t = ReweightingTransformer(protected_attributes=["gender"], method="inverse_frequency").fit(df)

    unseen = df.head(6).copy()
    unseen.loc[unseen.index[:3], "gender"] = "c"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = t.get_sample_weights(unseen)

    assert np.isnan(w[:3]).all(), f"a group unseen at fit was handed a weight: {w[:3]}"
    # ... and the rows that WERE fitted keep their measured weight: refusing the
    # whole call would throw away evidence the fit really produced.
    assert np.isfinite(w[3:]).all()
    assert w[3:] == pytest.approx(120.0 / (2 * 80), abs=1e-12)
    assert any("no weight was fitted" in m for m in _messages(caught))


def test_a_one_row_intersection_is_counted_not_silently_passed_through():
    """A 1-row group has std NaN under ddof=1 and `nan > 0` is False, so the row
    used to fall through transform() raw while the column was still reported as
    modified. transform() must count it and say so."""
    rng = np.random.default_rng(11)
    parts = [
        pd.DataFrame({"gender": g, "race": a, "score": rng.normal(mu, 2.0, 40)})
        for g, a, mu in [("f", "x", 9.6), ("f", "y", 14.8), ("m", "x", -0.5)]
    ]
    parts.append(pd.DataFrame({"gender": ["z"], "race": ["x"], "score": [999.0]}))
    df = pd.concat(parts, ignore_index=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = IntersectionalTransformer(protected_attributes=["gender", "race"], min_group_size=1)
        out = t.fit_transform(df)

    lone = [s for s in t._group_stats["score"].values() if s["count"] == 1]  # noqa: SLF001
    assert lone and np.isnan(lone[0]["std"]), "the fixture no longer reaches the branch"
    assert lone[0]["transformable"] is False
    assert out["score"].iloc[-1] == 999.0
    assert t.rows_not_transformed_.tolist() == [len(df) - 1]
    assert t.fit_result.fit_metrics["n_rows_not_transformed"] == 1
    assert any("UNTRANSFORMED" in m for m in _messages(caught))


def test_label_massager_on_one_group_says_there_was_nothing_to_compare():
    """A single observed group gives deprived == favored, so M is 0 and no label
    moves. That is a could-not-check, and it must be on the result rather than
    looking like a frame whose groups were already equal."""
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})
    y = (rng.random(60) < 0.5).astype(int)
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)
    assert lm.fit_result.fit_metrics["n_groups"] == 1
    assert lm.deprived_group_ == lm.favored_group_
    assert any("fewer than 2 groups" in w for w in lm.fit_result.warnings), (
        f"the fit published no refusal at all: {lm.fit_result.warnings}"
    )
    assert np.array_equal(lm.get_massaged_labels(df, y), y)
