"""BGL stage 3, batch preprocessing-1: the feature-engineering transformers.

Every public fit / transform / getter on
``vfairness.preprocessing.feature_engineering.transformers`` was executed on an
input where the quantity it reports does not exist, and on healthy input beside
it. Eight fabrications were found by execution on 2026-09-27 and are fixed here;
six refusals were already honest and are pinned so a refactor cannot undo them.

Found and fixed (each number below was printed by a run, not estimated):

  P1-1  ``ReweightingTransformer.fit`` counted its observed groups on the FIRST
        protected attribute while 'target_parity' keys its weights on the
        INTERSECTIONAL group. A frame whose first attribute had one level and
        whose joint key had two published ``n_groups_observed`` 1,
        ``between_group_balance_measured`` False and the refusal "the weights
        below are 1.0 by construction" over weights of 0.961538, 1.041667,
        1.029412 and 0.972222 that had genuinely equalized the joint-group
        positive rates (0.433333 and 0.400000 both to 0.416667).
        ``get_sample_weights`` repeated the same false claim over the same array.
        A measurement reported as a could-not-check is the same defect backwards.
  P1-2  ``ReweightingTransformer._group_key_series`` turned a MISSING protected
        value into the level "nan". 10 of 60 rows with a blank 'race' were coined
        into a group and given weights 1.291667 / 0.805556, no weight came back
        NaN, and ``fit_result.warnings == []``.
  P1-3  ``CorrelationReducer.transform`` encodes a protected level the fit never
        saw as the all-zero design row, which IS the reference category. Fitted
        on gender {a, b}, 20 rows of gender 'c' at proxy=40.0 came back
        39.922704, byte-identical to the same rows labelled 'a', where group 'b'
        would have been moved to 33.914867. No warning, no result field.
  P1-4  ``FairRepresentationTransformer._group_codes`` counted a missing
        protected value as a group. One observed level plus 10 blanks published
        ``n_groups`` 2 with ``fit_result.warnings == []``, and the adversary was
        trained to hide value-present from value-missing.
  P1-5  ``LabelMassager.get_massaged_labels`` was silent on a one-group frame: 0
        labels changed, ``n_labels_flipped_`` 0 and no warning from any surface,
        byte-identical to two groups whose rates were already equal.
  P1-6  ``ResidualTransformer.fit`` reported ``features_modified`` for columns it
        did not residualize. With a single-level 'gender' the largest |out - in|
        over both feature columns was 1.11e-16 while features_modified read
        ['f', 'g'].
  P1-7  ``FeatureSuppressor.fit`` never cleared ``_bin_edges`` / ``_mask_values``,
        so a REFIT that could not estimate them left the previous fit's standing.
        Measured: edges [-1.0671, 0.963, 2.985, 5.007, 7.0291] from fit 1, then a
        refit on a blank column recorded "could not be binned, so transform()
        returns them UNCHANGED" and transform() returned the constant 3.0 for all
        60 rows.
  P1-8  ``FeatureSuppressor`` masked with a mean of nan: features_modified named
        the column while transform() handed back the all-NaN column it was given.

Already honest, pinned here: DisparateImpactRemover.fit/.transform (rows with no
group, and an undersized group moving the shared target), Resampler's empty-frame
state reset, ResidualTransformer.transform's unseen-group disclosure,
CorrelationReducer.fit's three-state ``target_met``, and LabelMassager.fit's
imputed-feature disclosure.

Every refusal pin is paired with a measurable case, so a transformer that refused
everything would fail here too.
"""

# ruff: noqa: N999 - the batch id carries a hyphen and the file name has to match
# it, so this module is imported by path (pytest) rather than by an import
# statement. Renaming it would break the coordinator's lookup.

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.transformers import (
    CorrelationReducer,
    DisparateImpactRemover,
    FairRepresentationTransformer,
    FeatureSuppressor,
    LabelMassager,
    Resampler,
    ResidualTransformer,
    ReweightingTransformer,
)


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
    """Correlation ratio eta, written out here so a control does not copy the
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
# P1-1  ReweightingTransformer: count the groups the weights are keyed on
# ===========================================================================


def _one_marginal_two_joint():
    """'race' holds ONE level; 'race' x 'gender' holds two. 'target_parity' keys
    its weights on the joint group, so that is where the comparison happens."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "race": np.array(["a"] * n),
            "gender": np.array(["m", "f"] * (n // 2)),
        }
    )
    y = np.r_[np.ones(20, int), np.zeros(10, int), np.ones(5, int), np.zeros(25, int)]
    return df, y


def test_reweighting_counts_the_groups_its_weights_are_actually_keyed_on():
    """P1-1. Measured before the fix: n_groups_observed 1,
    between_group_balance_measured False, fit_result.warnings carrying "the
    weights below are 1.0 by construction ... This is a could-not-check", and
    get_sample_weights warning "these weights are 1.0 by construction" over the
    array [0.961538, 0.972222, 1.029412, 1.041667], which had equalized the
    joint-group positive rates 0.433333 and 0.400000 to 0.416667."""
    df, y = _one_marginal_two_joint()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(
            protected_attributes=["race", "gender"], method="target_parity"
        ).fit(df, y)
        w = t.get_sample_weights(df, y)

    metrics = t.fit_result.fit_metrics
    assert metrics["n_groups_observed"] == 2, (
        f"the count came from the wrong key: {metrics['n_groups_observed']}"
    )
    assert metrics["between_group_balance_measured"] is True
    assert "intersection" in metrics["group_key_counted"]
    assert t.fit_result.warnings == [], (
        f"a completed comparison was reported as a could-not-check: {t.fit_result.warnings}"
    )
    assert not any("1.0 by construction" in m for m in _messages(caught)), (
        f"the false refusal reached the caller: {_messages(caught)}"
    )

    # ... and the weights really did the work the refusal denied, recomputed here.
    assert float(np.ptp(w)) > 0.05, f"the weights are flat after all: {np.unique(w)}"
    key = df["race"].astype(str) + "|" + df["gender"].astype(str)
    rates = [
        float(np.average(y[(key == g).to_numpy()], weights=w[(key == g).to_numpy()]))
        for g in sorted(set(key))
    ]
    assert rates[0] == pytest.approx(rates[1], abs=1e-9)
    raw = [float(y[(key == g).to_numpy()].mean()) for g in sorted(set(key))]
    assert abs(raw[0] - raw[1]) > 0.01, "the fixture had no imbalance to remove"


def test_control_reweighting_still_refuses_a_single_group_of_the_key_it_uses():
    """CONTROL for P1-1: the refusal must survive where it is TRUE. One level of
    the only attribute makes every inverse-frequency weight exactly 1.0, and one
    level of the joint key does the same to the Kamiran and Calders weights."""
    rng = np.random.default_rng(1)
    flat = pd.DataFrame({"f": rng.normal(size=60), "race": ["a"] * 60, "gender": ["m"] * 60})
    y = np.r_[np.ones(30, int), np.zeros(30, int)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(
            protected_attributes=["race", "gender"], method="target_parity"
        ).fit(flat, y)
        w = t.get_sample_weights(flat, y)

    assert t.fit_result.fit_metrics["n_groups_observed"] == 1
    assert t.fit_result.fit_metrics["between_group_balance_measured"] is False
    assert any("1 observed group" in v for v in t.fit_result.warnings)
    assert any("could-not-check" in v for v in t.fit_result.warnings)
    assert any("1.0 by construction" in m for m in _messages(caught))
    assert np.allclose(w, 1.0)


def test_a_missing_protected_value_is_not_a_reweighting_group():
    """P1-2. Measured before the fix: the weight keys included ('nan', 1) and
    ('nan', 0) at 1.291667 and 0.805556, n_groups_observed 2 while THREE groups
    carried weights, 0 rows came back NaN, and fit_result.warnings == []."""
    rng = np.random.default_rng(1)
    race = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    race[:10] = np.nan
    df = pd.DataFrame({"f1": rng.normal(size=60), "race": race})
    y = (rng.random(60) < 0.5).astype(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(protected_attributes=["race"], method="target_parity").fit(df, y)
        w = t.get_sample_weights(df, y)

    keys = list(t.fit_result.fit_metrics["group_weights"])
    assert not any("nan" in k for k in keys), f"missingness was coined into a group: {keys}"
    assert t.fit_result.fit_metrics["n_rows_without_a_recorded_group"] == 10
    assert np.isnan(w[:10]).all(), f"rows with no group were handed a weight: {w[:10]}"
    assert np.isfinite(w[10:]).all(), "the rows that DO have a group lost their weight"
    assert any("belong to no known group" in v for v in t.fit_result.warnings)
    assert any("no weight was fitted" in m for m in _messages(caught))


# ===========================================================================
# P1-3  CorrelationReducer.transform: an unfitted level is not the reference
# ===========================================================================


def test_correlation_reducer_does_not_silently_reduce_an_unfitted_group():
    """P1-3. Measured before the fix, method='residualize' fitted on gender
    {a, b}: 20 rows of gender 'c' at proxy=40.0 came back 39.922704, byte
    identical to the same rows labelled 'a', while group 'b' would have been
    moved to 33.914867. warnings == [] and there was no result channel at all."""
    rng = np.random.default_rng(0)
    gender = np.array(["a"] * 60 + ["b"] * 60)
    df = pd.DataFrame(
        {"proxy": np.where(gender == "a", 0.0, 6.0) + rng.normal(0, 1.0, 120), "gender": gender}
    )
    r = CorrelationReducer(
        protected_attributes=["gender"], method="residualize", preserve_variance=False
    ).fit(df)

    def out_for(level):
        frame = pd.DataFrame({"proxy": np.full(20, 40.0), "gender": [level] * 20})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = r.transform(frame)["proxy"].to_numpy()
        # transform_result describes the LAST call, so it is taken here rather
        # than after the comparison calls below have overwritten it.
        return values, _messages(caught), dict(r.transform_result)

    unseen, caught_unseen, result_unseen = out_for("c")
    reference, _, _ = out_for("a")
    other, _, _ = out_for("b")

    # The fixture reaches the branch: the unfitted level gets the REFERENCE
    # group's correction, and the two fitted groups differ by the real gap.
    assert np.allclose(unseen, reference)
    assert abs(float(reference[0] - other[0])) > 5.0
    assert any("the fit never saw" in m for m in caught_unseen), (
        f"a row corrected as the reference group was not disclosed: {caught_unseen}"
    )
    assert result_unseen["n_rows_with_an_unfitted_level"] == 20
    assert result_unseen["unfitted_levels"] == {"gender": ["c"]}


def test_control_correlation_reducer_says_nothing_about_a_frame_it_fitted():
    """CONTROL for P1-3: the disclosure must not fire for levels the fit DID see,
    and the reduction must still work."""
    df, _ = _healthy()
    r = CorrelationReducer(protected_attributes=["gender"], method="residualize").fit(df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = r.transform(df)

    assert _messages(caught) == [], f"a fully fitted frame was disclosed anyway: {caught}"
    assert r.transform_result["n_rows_with_an_unfitted_level"] == 0
    assert r.transform_result["n_rows_without_a_recorded_group"] == 0
    assert _eta(df["proxy"].to_numpy(), df["gender"].to_numpy()) > 0.9
    assert _eta(out["proxy"].to_numpy(), df["gender"].to_numpy()) < 1e-6


# ===========================================================================
# P1-4  FairRepresentationTransformer: missingness is not a protected group
# ===========================================================================


def test_fair_representation_does_not_count_a_missing_value_as_a_group():
    """P1-4. Measured before the fix on 60 rows whose 'gender' held ONE observed
    level plus 10 blanks: fit_metrics['n_groups'] == 2, fit_result.warnings == []
    and no Python warning, byte-identical to a genuine two-group run, while the
    adversary was trained to hide value-present from value-missing."""
    rng = np.random.default_rng(11)
    gender = np.array(["a"] * 60, dtype=object)
    gender[:10] = np.nan
    df = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60), "gender": gender})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=5
        ).fit(df)

    metrics = f.fit_result.fit_metrics
    assert metrics["n_groups"] == 1, f"missingness was counted as a group: {metrics['n_groups']}"
    assert metrics["n_rows_without_a_recorded_group"] == 10
    assert metrics["n_rows_in_the_adversary_target"] == 50
    assert any("NO fairness constraint was applied" in m for m in _messages(caught))
    assert any("belong to no known group" in v for v in f.fit_result.warnings)
    # The codes themselves: -1 is "no group", never a level.
    assert sorted(np.unique(f._group_codes(df)).tolist()) == [-1, 0]  # noqa: SLF001


def test_control_fair_representation_keeps_a_real_two_group_comparison():
    """CONTROL for P1-4: blanks beside TWO real levels must not collapse the
    count to one, and the encoder must still encode."""
    rng = np.random.default_rng(12)
    gender = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    gender[:5] = np.nan
    df = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60), "gender": gender})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=5
        ).fit(df)
        z = f.transform(df)

    assert f.fit_result.fit_metrics["n_groups"] == 2
    assert f.fit_result.fit_metrics["n_rows_in_the_adversary_target"] == 55
    assert not any("NO fairness constraint" in m for m in _messages(caught)), (
        "a real two-group fit was refused"
    )
    assert np.isfinite(z.to_numpy()).all()
    assert float(np.ptp(z["rep_0"].to_numpy())) > 0.1


# ===========================================================================
# P1-5  LabelMassager.get_massaged_labels: zero flips is not zero disparity
# ===========================================================================


def test_label_massager_says_so_when_zero_flips_means_nothing_was_compared():
    """P1-5. Measured before the fix on 60 single-group rows with two usable
    features: the ranker trained (fit_metrics['ranker'] == 'logistic'), 0 labels
    changed, n_labels_flipped_ / n_promoted_ / n_demoted_ all 0, and
    get_massaged_labels emitted NO warning at all."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60), "gender": ["a"] * 60})
    y = (rng.random(60) < 0.4).astype(int)
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)

    # The fixture reaches the branch under test rather than the ranker branch.
    assert lm.fit_result.fit_metrics["ranker"] == "logistic"
    assert lm.fit_result.fit_metrics["n_groups"] == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_new = lm.get_massaged_labels(df, y)

    assert np.array_equal(y_new, y)
    assert lm.n_labels_flipped_ == 0
    assert any("could-not-check" in m for m in _messages(caught)), (
        f"an unchanged label array reached the caller with no signal: {_messages(caught)}"
    )
    assert any("1 protected group" in m for m in _messages(caught))


def test_control_label_massager_stays_silent_when_two_groups_really_tie():
    """CONTROL for P1-5: with two groups at exactly the same positive rate,
    min() and max() both return the first key, so deprived == favored there too.
    That is a MEASUREMENT of equality and must not be disclosed as a refusal."""
    rng = np.random.default_rng(4)
    gender = np.array(["a"] * 30 + ["b"] * 30)
    y = np.r_[np.ones(12, int), np.zeros(18, int), np.ones(12, int), np.zeros(18, int)]
    df = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60), "gender": gender})
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)

    assert lm.deprived_group_ == lm.favored_group_, "the fixture no longer reaches the tie"
    assert [float(y[gender == g].mean()) for g in ("a", "b")] == [0.4, 0.4]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_new = lm.get_massaged_labels(df, y)

    assert np.array_equal(y_new, y)
    assert _messages(caught) == [], f"a measured tie was reported as a refusal: {caught}"

    # ... and a real gap is still massaged, silently.
    dfh, yh = _healthy()
    lmh = LabelMassager(protected_attributes=["gender"]).fit(dfh, yh)
    with warnings.catch_warnings(record=True) as caught_h:
        warnings.simplefilter("always")
        yh_new = lmh.get_massaged_labels(dfh, yh)
    assert int((yh_new != yh).sum()) > 0
    assert _messages(caught_h) == []


# ===========================================================================
# P1-6  ResidualTransformer.fit: a column it did not residualize
# ===========================================================================


def test_residual_transformer_does_not_report_a_feature_it_did_not_residualize():
    """P1-6. Measured before the fix on 60 rows of a single-level 'gender':
    features_modified == ['f', 'g'] while the largest |out - in| across both
    columns was 1.1102230246251565e-16, so no residualization happened and no
    channel said the conditioning had no contrast."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"f": rng.normal(size=60), "g": rng.normal(size=60), "gender": ["a"] * 60})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)
        out = rt.transform(df)

    # The fixture reaches the branch: one conditional mean, and it IS the global one.
    assert list(rt._group_means["f"]) == ["a"]  # noqa: SLF001
    assert float(np.max(np.abs(out["f"].to_numpy() - df["f"].to_numpy()))) < 1e-12

    assert rt.fit_result.features_modified == [], (
        f"a no-op residualization was reported as a modification: {rt.fit_result.features_modified}"
    )
    assert any("no between-group difference to remove" in v for v in rt.fit_result.warnings)
    assert any("fewer than 2 groups" in m for m in _messages(caught))


def test_control_residual_transformer_still_reports_the_columns_it_did_residualize():
    """CONTROL for P1-6: two groups, a real gap removed, and the column named as
    modified. A fix that emptied features_modified would pass P1-6 and destroy
    the finding the report carries."""
    df, _ = _healthy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)
        out = rt.transform(df)

    assert rt.fit_result.features_modified == ["proxy", "clean", "age"]
    assert _eta(df["proxy"].to_numpy(), df["gender"].to_numpy()) > 0.9
    assert _eta(out["proxy"].to_numpy(), df["gender"].to_numpy()) < 1e-9
    assert rt.fit_result.warnings == []
    assert _messages(caught) == []


# ===========================================================================
# P1-7 / P1-8  FeatureSuppressor: a suppression that did not happen
# ===========================================================================


def _good_and_blank():
    rng = np.random.default_rng(4)
    gender = np.array(["a"] * 30 + ["b"] * 30)
    good = pd.DataFrame(
        {"proxy": np.where(gender == "a", 0.0, 6.0) + rng.normal(0, 0.5, 60), "gender": gender}
    )
    blank = good.copy()
    blank["proxy"] = np.nan
    return good, blank


def test_feature_suppressor_refit_does_not_apply_the_previous_bin_edges():
    """P1-7. Measured before the fix: fit 1 stored edges
    [-1.0671, 0.963, 2.985, 5.007, 7.0291]; the refit on a blank 'proxy'
    recorded "could not be binned, so transform() returns them UNCHANGED",
    features_modified == [] and features_not_suppressed == ['proxy'], and then
    transform() returned the constant 3.0 for all 60 rows by digitizing against
    those stale edges."""
    good, blank = _good_and_blank()
    fs = FeatureSuppressor(
        protected_attributes=["gender"], strategy="bin", features_to_suppress=["proxy"], n_bins=4
    ).fit(good)
    # The first fit really did estimate a binning, and it really bins.
    assert "proxy" in fs._bin_edges  # noqa: SLF001
    assert np.unique(fs.transform(good)["proxy"].to_numpy()).size > 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs.fit(blank)

    assert "proxy" not in fs._bin_edges, (  # noqa: SLF001
        "the refit kept the previous fit's bin edges"
    )
    out = fs.transform(blank)
    assert np.isnan(out["proxy"].to_numpy()).all(), (
        "a binning estimated from data this fit never saw was applied anyway"
    )
    assert fs.fit_result.features_modified == []
    assert fs.fit_result.fit_metrics["features_not_suppressed"] == ["proxy"]
    assert any("could not be suppressed" in m for m in _messages(caught))


def test_feature_suppressor_does_not_report_a_mask_it_could_not_compute():
    """P1-8. Measured before the fix: _mask_values['proxy'] was nan (the mean of
    a column with no observation), transform() handed back the all-NaN column it
    was given, and features_modified still read ['proxy'] with
    features_not_suppressed == []."""
    good, blank = _good_and_blank()
    fs = FeatureSuppressor(
        protected_attributes=["gender"], strategy="mask", features_to_suppress=["proxy"]
    ).fit(blank)

    assert "proxy" not in fs._mask_values, (  # noqa: SLF001
        f"a nan was stored as a mask value: {fs._mask_values}"  # noqa: SLF001
    )
    assert fs.fit_result.features_modified == []
    assert fs.fit_result.fit_metrics["features_not_suppressed"] == ["proxy"]
    assert any("no mask value to write" in v for v in fs.fit_result.warnings)
    out = fs.transform(blank)
    assert np.isnan(out["proxy"].to_numpy()).all()


def test_control_feature_suppressor_still_masks_a_column_it_can_measure():
    """CONTROL for P1-7 and P1-8: the same call on a column with values must
    still be suppressed, named as modified, and silent."""
    good, _ = _good_and_blank()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs = FeatureSuppressor(
            protected_attributes=["gender"], strategy="mask", correlation_threshold=0.3
        ).fit(good)
        out = fs.transform(good)

    assert fs.fit_result.features_modified == ["proxy"]
    assert fs.fit_result.fit_metrics["features_not_suppressed"] == []
    assert float(np.ptp(out["proxy"].to_numpy())) == 0.0, "the mask did not flatten the column"
    assert out["proxy"].to_numpy()[0] == pytest.approx(float(good["proxy"].mean()), abs=1e-12)
    # np.ptp, never an eta recomputed here: the sum of squared deviations of 60
    # copies of 2.9571715711488675 is ~1e-30 of ulp noise rather than 0, and the
    # ratio of two such numbers is arbitrary (it reads 1.0 for this fixture). The
    # library's own measurement carries the same guard and is read instead.
    assert _eta(good["proxy"].to_numpy(), good["gender"].to_numpy()) > 0.9
    assert fs.fit_result.correlation_before["gender"] > 0.9
    assert fs.fit_result.correlation_after["gender"] < 1e-9
    assert fs.fit_result.warnings == []
    assert _messages(caught) == []


# ===========================================================================
# Refusals that were already honest, pinned so a refactor cannot undo them
# ===========================================================================


def test_disparate_impact_remover_excludes_a_row_with_no_group_and_says_so():
    """Already honest. 6 of 60 rows with a blank 'race' are excluded from every
    fitted quantile curve, returned UNREPAIRED, and counted on both fit_result
    and transform_result. Pinned because ``astype(str)`` would silently make
    them the group "nan" again, which is what happened in this class before."""
    rng = np.random.default_rng(2)
    race = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    race[:6] = np.nan
    df = pd.DataFrame({"f": rng.normal(size=60), "race": race})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = DisparateImpactRemover(protected_attributes=["race"]).fit(df)
        out = t.transform(df)

    assert t.fit_result.fit_metrics["n_groups"] == 2
    assert t.fit_result.fit_metrics["groups"] == ["a", "b"]
    assert t.fit_result.fit_metrics["n_rows_without_a_recorded_group"] == 6
    assert t.transform_result["n_rows_unrepaired"] == 6
    assert t.transform_result["n_rows_without_a_recorded_group"] == 6
    assert np.allclose(out["f"].to_numpy()[:6], df["f"].to_numpy()[:6])
    assert not np.allclose(out["f"].to_numpy()[6:], df["f"].to_numpy()[6:]), (
        "the rows that DO have a group were not repaired either"
    )
    assert any("belong to no known group" in m for m in _messages(caught))


def test_disparate_impact_remover_names_an_undersized_group_estimate():
    """Already honest. One 'b' row gives np.quantile a flat curve at all 101
    knots, and np.median of two curves is their mean, so a single observation
    moves the repair target for every row: measured, the 59 'a' rows went from
    mean -0.0293 to 15.0058. Pinned as a could-not-check, not a repair."""
    rng = np.random.default_rng(2)
    race = np.array(["a"] * 59 + ["b"])
    df = pd.DataFrame({"f": np.r_[rng.normal(0, 1, 59), 30.0], "race": race})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = DisparateImpactRemover(protected_attributes=["race"]).fit(df)
        out = t.transform(df)

    assert t.fit_result.fit_metrics["features_repaired_on_unreliable_estimate"] == ["f"]
    assert t.fit_result.fit_metrics["n_per_group"]["f"] == {"a": 59, "b": 1}
    assert any("no distribution to estimate" in m for m in _messages(caught))
    # The disclosure is not decorative: the target really did move that far.
    assert float(out["f"].to_numpy()[:59].mean()) > 10.0


def test_resampler_clears_the_previous_calls_numbers_on_an_empty_frame():
    """Already honest. An empty call resets the state instead of leaving the
    previous resample's numbers standing, and max_replication_factor is None
    rather than 1.0, which would read as a measured perfect run."""
    df, y = _healthy(n=60)
    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        rs.get_resampled_data(df, y)
    assert rs.resample_result["n_rows_out"] > 0
    assert rs.resample_result["max_replication_factor"] is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_out, y_out = rs.get_resampled_data(df.iloc[:0], y[:0])

    assert len(x_out) == 0 and len(y_out) == 0
    assert rs.resample_result["n_rows_out"] == 0
    assert rs.resample_result["cell_sizes_after"] == {}
    assert rs.resample_result["max_replication_factor"] is None
    assert "could_not_check" in rs.resample_result
    assert any("nothing was resampled" in m for m in _messages(caught))
    assert any("nothing was resampled" in v for v in rs.fit_result.warnings)


def test_residual_transformer_transform_names_a_group_it_never_fitted():
    """Already honest, and unpinned until now. A group value absent from fit has
    no conditional mean, so ``.get(group, global_mean)`` leaves its rows exactly
    as they came in: measured, 10 rows at 40.0 came back 40.0. The pass-through
    is disclosed rather than presented as a residual."""
    rng = np.random.default_rng(6)
    gender = np.array(["a"] * 30 + ["b"] * 30)
    df = pd.DataFrame(
        {"f": np.where(gender == "a", 0.0, 6.0) + rng.normal(0, 0.5, 60), "gender": gender}
    )
    rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)

    new = pd.DataFrame({"f": np.full(10, 40.0), "gender": ["c"] * 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = rt.transform(new)

    assert np.allclose(out["f"].to_numpy(), 40.0), "the fixture no longer reaches the branch"
    assert any("were not seen at fit" in m for m in _messages(caught))
    assert any("un-residualized" in m for m in _messages(caught))


def test_correlation_reducer_fit_leaves_an_unmeasurable_attribute_unmeasured():
    """Already honest. A single-level attribute has no contrast, so eta is 0.0 by
    construction rather than by measurement: the attribute is absent from
    correlation_before / after, target_met is None (not True), and the reason is
    on the result. Pinned because ``bool(nan <= target)`` is False, which is how
    a NaN used to be graded "measured, target not met"."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 60})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = CorrelationReducer(protected_attributes=["gender"]).fit(df)

    res = r.fit_result
    assert res.correlation_before == {} and res.correlation_after == {}
    assert res.correlation_reduction == {}
    assert res.fit_metrics["target_met"] == {"gender": None}
    assert res.fit_metrics["attributes_not_measured"] == ["gender"]
    assert any("target_met is None" in v for v in res.warnings)
    assert any("only 1 observed level" in m for m in _messages(caught))

    # ... and a real two-group frame still earns a True verdict, so this is not
    # a transformer that refuses everything.
    healthy, _ = _healthy()
    ok = CorrelationReducer(protected_attributes=["gender"], target_correlation=0.1).fit(healthy)
    assert ok.fit_result.fit_metrics["target_met"] == {"gender": True}
    assert ok.fit_result.fit_metrics["attributes_not_measured"] == []


def test_label_massager_fit_puts_the_imputed_ranker_values_on_the_result():
    """Already honest. Ranker features are filled with 0.0, so a flip decision
    for a row with a blank feature rests on a substituted value. Measured on 30
    blanked 'clean' cells: the count reaches fit_metrics AND fit_result.warnings,
    not only an instance attribute."""
    df, y = _healthy()
    df = df.copy()
    df.loc[df.index[:30], "clean"] = np.nan

    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)

    assert lm.fit_result.fit_metrics["n_features_imputed"] == 30
    assert any("filled with 0.0" in v for v in lm.fit_result.warnings)
    # The fit still did its job: the flips are the measured total, not a plan.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_new = lm.get_massaged_labels(df, y)
    assert lm.n_labels_flipped_ == int((y_new != y).sum()) > 0
