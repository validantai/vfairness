"""BGL5 pins for batch A-preprocessing-1: the nine overturned grades, closed.

Every grade the BGL4 audit overturned in
``/tmp/claude-501/bgl/results/audit-A-preprocessing-1.json`` is fixed in
``preprocessing/feature_engineering/transformers.py`` and
``preprocessing/feature_engineering/analyzer.py``, and pinned here. Each number in
a docstring below was printed by running the unit on this repo on 2026-09-27,
before and after.

FOUR OF THE NINE WERE ONE MECHANISM: a guard that reads the FIT while the defect
arrives at the CALL, or a guard whose condition structurally cannot fire.

    LabelMassager.get_massaged_labels   both guards read the fit, and
                                        _planned_flip_count returns 0 as soon as
                                        the CALL frame carries no deprived or no
                                        favored row
    ResidualTransformer.fit/.transform   fit stored nan in _fitted_groups, so
                                        ``g not in self._fitted_groups`` could
                                        never be True for a row with no protected
                                        value
    Resampler.get_resampled_data         astype(str) made missingness the group
                                        "nan", which also made the single-group
                                        refusal unreachable
    FeatureSuppressor.fit/.transform     the refusal was written
                                        ``elif self.strategy == "mask"``, so
                                        strategy='noise' had no degeneracy check

Each pin has an over-correction control asserting that healthy input still gets
its real measured number, because a fix that refuses everything passes every
refusal test and destroys the library.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.analyzer import FeatureEngineeringAnalyzer
from vfairness.preprocessing.feature_engineering.transformers import (
    FairRepresentationTransformer,
    FeatureSuppressor,
    LabelMassager,
    Resampler,
    ResidualTransformer,
    ReweightingTransformer,
)


def _messages(caught) -> list:
    return [str(w.message) for w in caught]


def _two_group(n: int = 200, seed: int = 0):
    """The healthy frame the audit fitted on: two real groups, no missing value."""
    rng = np.random.default_rng(seed)
    gender = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    df = pd.DataFrame(
        {
            "proxy": np.where(gender == "a", 0.0, 6.0) + rng.normal(0, 1.0, n),
            "clean": rng.normal(0, 1.0, n),
            "gender": gender,
        }
    )
    y = (rng.random(n) < np.where(gender == "a", 0.25, 0.75)).astype(int)
    return df, y


# ===========================================================================
# A1  LabelMassager.get_massaged_labels: the guard now reads the CALL frame
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "level"),
    [
        ("a level the fit never saw", "c"),
        ("only the favored group", "b"),
        ("no group at all", None),
    ],
)
def test_label_massager_names_a_call_frame_that_carries_no_comparable_group(label, level):
    """Before (measured 2026-09-27): fitted on the 200-row two-group frame
    (n_groups 2, ranker 'logistic', deprived 'a', favored 'b', 60 labels flipped at
    fit), then called on 40 rows of gender 'c' / all 'b' / all missing, the array
    came back byte-identical, ``n_labels_flipped_`` 0 and ``warnings == []`` in all
    three cases. ``_planned_flip_count`` returns 0 when ``n_dep == 0 or n_fav == 0``
    and the only guards were ``self._model is None`` and
    ``self._n_groups_fitted < 2``, both properties of the FIT.

    After: each of the three warns, naming how many rows of the deprived and of the
    favored group THIS frame carries ("0 row(s) of the deprived group 'a' and 0
    row(s) of the favored group 'b'", "0 ... and 40 ...", and "40 of 40 row(s)
    carry no protected value at all"), and says n_labels_flipped_ == 0 is a
    could-not-check for this frame.
    """
    df, y = _two_group()
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)
    assert lm.fit_result.fit_metrics["n_groups"] == 2, "the premise is gone"

    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "proxy": np.full(40, 3.0),
            "clean": rng.normal(size=40),
            "gender": [level] * 40 if level is not None else [np.nan] * 40,
        }
    )
    yy = (rng.random(40) < 0.2).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = lm.get_massaged_labels(frame, yy)

    assert np.array_equal(out, yy), "the fixture no longer reaches the branch"
    assert lm.n_labels_flipped_ == 0
    said = _messages(caught)
    assert any("could-not-check" in m for m in said), (
        f"0 flips over a comparison that never happened reached the caller silently: {said}"
    )
    assert any("row(s) of the deprived group" in m for m in said), (
        f"the disclosure does not say which side was absent: {said}"
    )
    if level is None:
        assert any("40 of 40 row(s) carry no protected value" in m for m in said), said


def test_control_label_massager_still_flips_the_labels_it_can_compare():
    """OVER-CORRECTION CONTROL. A guard that fires on every call would destroy the
    transformer, so the two healthy cases keep their real measured answer.

    Measured 2026-09-27, after the fix: called on the frame it was fitted on, the
    massager still flips exactly 60 labels ((y_new != y).sum() == 60,
    n_labels_flipped_ == 60) and emits NO warning. And two groups whose positive
    rates genuinely TIE (100 'a' and 100 'b', both at 0.4) return y unchanged with
    NO warning either: the guard counts rows of the deprived and the favored group,
    not ``deprived == favored``, which a real tie also makes true.
    """
    df, y = _two_group()
    lm = LabelMassager(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = lm.get_massaged_labels(df, y)
    assert int((out != y).sum()) == 60
    assert lm.n_labels_flipped_ == 60
    assert _messages(caught) == [], "a measured relabelling was caveated"

    rng = np.random.default_rng(3)
    gender = np.array(["a"] * 100 + ["b"] * 100)
    tie = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200), "gender": gender})
    y_tie = np.r_[np.repeat([1, 0], [40, 60]), np.repeat([1, 0], [40, 60])]
    lt = LabelMassager(protected_attributes=["gender"]).fit(tie, y_tie)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_tie = lt.get_massaged_labels(tie, y_tie)
    assert np.array_equal(out_tie, y_tie), "the tie fixture drifted"
    assert _messages(caught) == [], (
        f"a genuine 0.4 / 0.4 tie was reported as a could-not-check: {_messages(caught)}"
    )


# ===========================================================================
# A2  ResidualTransformer.fit and .transform: nan is not a fitted group
# ===========================================================================


def _residual_frame():
    rng = np.random.default_rng(6)
    gender = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    gender[:10] = np.nan
    df = pd.DataFrame(
        {
            "f": np.where(pd.isna(gender), 40.0, np.where(gender == "a", 0.0, 6.0))
            + rng.normal(0, 0.5, 60),
            "gender": gender,
        }
    )
    return df


def test_residual_transformer_counts_the_rows_it_could_not_residualize():
    """Before (measured 2026-09-27) on 60 rows whose 'gender' held two real levels
    plus 10 missing values: ``_group_means['f']`` keys ['a', 'b'] (missingness
    correctly NOT a group) but ``_fitted_groups`` [nan, 'a', 'b'],
    ``features_modified`` ['f'], ``fit_result.warnings == []``, zero Python
    warnings, and max |out - in| over the 10 groupless rows exactly 0.0 while the
    other 50 moved by up to 9.727115410387928.

    ``transform`` computes ``unseen = [g for g in pd.unique(df[attr]) if g not in
    self._fitted_groups]``, and nan WAS in ``_fitted_groups``, so the refusal was
    structurally unreachable for exactly the rows with no conditional mean.

    After: ``_fitted_groups`` == ['a', 'b'], fit carries "10 of 60 row(s) have a
    missing value in 'gender' ... E[X | A] does not exist for them ... Treat those
    rows as could-not-check, not as residuals" on ``fit_result.warnings`` AND as a
    Python warning, and transform warns "10 of 60 row(s) have no value for
    'gender' ... they pass through un-residualized". The columns stay in
    features_modified because the 50 rows that DO carry a group were genuinely
    residualized; withdrawing them would delete that real measurement.
    """
    df = _residual_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)
        out = rt.transform(df)

    raw, got = df["f"].to_numpy(), out["f"].to_numpy()
    assert np.array_equal(got[:10], raw[:10]), "the fixture no longer reaches the branch"
    assert not np.allclose(got[10:], raw[10:]), "nothing was residualized at all"

    assert list(rt._fitted_groups) == ["a", "b"], (
        f"a value with no conditional mean is recorded as a fitted group: {rt._fitted_groups}"
    )
    assert sorted(rt._group_means["f"]) == ["a", "b"]
    assert rt.fit_result.features_modified == ["f"]

    fit_said = list(rt.fit_result.warnings)
    assert any("no known group" in m and "10 of 60" in m for m in fit_said), (
        f"the fit record does not count the rows it could not residualize: {fit_said}"
    )
    said = _messages(caught)
    assert any("no conditional mean" in m and "10 of 60" in m for m in said), (
        f"transform returned 10 of 60 rows un-residualized inside the residual column "
        f"with nothing said: {said}"
    )


def test_control_residual_transformer_still_residualizes_a_complete_frame():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on the 200-row
    two-group frame with NO missing value: ``features_modified`` ['proxy', 'clean'],
    ``_fitted_groups`` ['a', 'b'], ``correlation_before['gender']``
    0.4993901205978534 against ``correlation_after['gender']``
    2.5949332499102044e-16, the between-group mean gap on 'proxy' 5.86833289233845
    before and exactly 0.0 after, ``fit_result.warnings == []`` and zero Python
    warnings. The new disclosure must not fire for a frame that has no groupless row.
    """
    df, _ = _two_group()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)
        out = rt.transform(df)

    assert rt.fit_result.features_modified == ["proxy", "clean"]
    assert rt.fit_result.correlation_before["gender"] == pytest.approx(0.4993901, abs=1e-6)
    assert rt.fit_result.correlation_after["gender"] < 1e-9
    gap_before = float(abs(df.groupby("gender")["proxy"].mean().diff().iloc[-1]))
    after = out.assign(g=df["gender"].to_numpy())
    gap_after = float(abs(after.groupby("g")["proxy"].mean().diff().iloc[-1]))
    assert gap_before == pytest.approx(5.8683329, abs=1e-6)
    assert gap_after < 1e-12
    assert list(rt.fit_result.warnings) == []
    assert _messages(caught) == [], f"a complete frame was caveated: {_messages(caught)}"


# ===========================================================================
# A3  FeatureSuppressor.fit and .transform: strategy='noise' has a guard now
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "column"),
    [("a constant column", np.full(60, 7.0)), ("a column with no value", np.full(60, np.nan))],
)
def test_feature_suppressor_noise_refuses_a_scale_that_perturbs_nothing(label, column):
    """Before (measured 2026-09-27) with strategy='noise' and
    features_to_suppress=['proxy']: a CONSTANT column of 7.0 gave
    ``features_modified == ['proxy']``, ``features_not_suppressed == []`` and output
    byte-identical to input (the scale is 0.1 * nanstd == 0.0); an all-NaN column
    gave the same record with numpy's "Degrees of freedom <= 0 for slice" as the
    only signal. The P1-8 refusal is written ``elif self.strategy == "mask"``, so it
    could not fire for 'noise'.

    After: both record ``features_modified == []``,
    ``features_not_suppressed == ['proxy']`` and the "could not be suppressed with
    strategy='noise' ... transform() returns them UNCHANGED" disclosure on
    ``fit_result.warnings`` and as a Python warning, and transform adds nothing
    rather than a zero-width or NaN perturbation.
    """
    df = pd.DataFrame({"proxy": column, "gender": np.array(["a"] * 30 + ["b"] * 30)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs = FeatureSuppressor(
            protected_attributes=["gender"], strategy="noise", features_to_suppress=["proxy"]
        ).fit(df)
        out = fs.transform(df)

    raw = df["proxy"].to_numpy(dtype=float)
    got = out["proxy"].to_numpy(dtype=float)
    unchanged = np.array_equal(np.nan_to_num(raw, nan=-12345.0), np.nan_to_num(got, nan=-12345.0))
    assert unchanged, "the fixture no longer reaches the branch"

    assert fs.fit_result.features_modified == [], (
        f"a suppression that changed nothing was published as a modification: "
        f"{fs.fit_result.features_modified}"
    )
    assert fs.fit_result.fit_metrics["features_not_suppressed"] == ["proxy"]
    assert "proxy" not in fs._noise_scales
    disclosed = _messages(caught) + list(fs.fit_result.warnings)
    assert any("could not be suppressed" in m for m in disclosed), disclosed


def test_control_feature_suppressor_noise_still_perturbs_a_column_it_can_measure():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on the 200-row
    two-group frame: strategy='noise' on 'proxy' still reports
    ``features_modified == ['proxy']`` and ``features_not_suppressed == []``, the
    fitted scale is exactly 0.1 * nanstd('proxy') == 0.3086885727443676, and the
    transformed column differs from the input in every row. Only the
    "correlation_after not measured" note for 'noise' is carried, which predates
    this fix.
    """
    df, _ = _two_group()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs = FeatureSuppressor(
            protected_attributes=["gender"], strategy="noise", features_to_suppress=["proxy"]
        ).fit(df)
        out = fs.transform(df)

    assert fs.fit_result.features_modified == ["proxy"]
    assert fs.fit_result.fit_metrics["features_not_suppressed"] == []
    expected = 0.1 * float(np.nanstd(df["proxy"].to_numpy(dtype=float)))
    assert expected == pytest.approx(0.3086885727443676, rel=1e-9)
    assert fs._noise_scales["proxy"] == pytest.approx(expected, rel=1e-12)
    moved = np.abs(out["proxy"].to_numpy() - df["proxy"].to_numpy())
    assert (moved > 0).all(), "the noise was not applied at all"
    assert not any("could not be suppressed" in m for m in _messages(caught))


# ===========================================================================
# A4  Resampler.get_resampled_data: missingness is not a group to balance
# ===========================================================================


def test_resampler_does_not_balance_against_a_group_made_of_missing_values():
    """Before (measured 2026-09-27) on 60 rows whose 'gender' held ONE observed
    level 'a' plus 10 missing values: ``cell_sizes_before``
    {'(nan, label=1)': 10, '(a, label=1)': 20, '(a, label=0)': 30},
    ``cell_sizes_after`` all three at 30, ``group_totals_after``
    {'nan': 30, 'a': 60}, ``n_rows_out`` 90 of which 30 carried a missing gender
    (20 of them copies of rows whose group is unknown), ``could_not_check`` None,
    and NO single-group disclosure, because ``observed_group_keys`` came from the
    same ``astype(str)`` values and so had length 2.

    After: the cell and group keys are ['(a, label=1)', '(a, label=0)'] and {'a':
    60}, ``n_rows_without_a_recorded_group`` is 10, ``n_rows_out`` is 70 (the 10
    groupless rows are returned once each, unbalanced, rather than deleted), and
    "between-group balance was NOT measured: only one protected group (['a'])"
    fires on the frame it was written for.
    """
    rng = np.random.default_rng(0)
    gender = np.array(["a"] * 60, dtype=object)
    gender[:10] = np.nan
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": gender})
    y = np.r_[np.ones(30, int), np.zeros(30, int)]

    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_out, y_out = rs.get_resampled_data(df, y)

    res = rs.resample_result
    keys = (
        list(res["cell_sizes_before"])
        + list(res["cell_sizes_after"])
        + list(res["group_totals_after"])
    )
    assert not any("nan" in str(k) for k in keys), (
        f"missingness was coined into a resampled group: {keys}"
    )
    assert res["group_totals_after"] == {"a": 60}
    assert res["n_rows_without_a_recorded_group"] == 10
    assert res["n_rows_out"] == 70 == len(x_out) == len(y_out)
    assert int(x_out["gender"].isna().sum()) == 10, "the groupless rows were deleted, not disclosed"

    disclosed = _messages(caught) + list(rs.fit_result.warnings)
    assert any("only one protected group" in m for m in disclosed), (
        f"a single-group frame was resampled as if it had two: {disclosed}"
    )
    assert any("no known group" in m for m in disclosed), disclosed


def test_control_resampler_still_balances_a_frame_with_two_real_groups():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on 60 rows,
    'gender' 30 'a' and 30 'b', labels alternating: all four cells 15 before and 15
    after, ``group_totals_after`` {'a': 30, 'b': 30}, ``n_rows_out`` 60,
    ``n_duplicated_rows`` 0, ``n_effective_rows`` 60,
    ``max_replication_factor`` 1.0, ``n_rows_without_a_recorded_group`` 0,
    ``fit_result.warnings == []`` and zero Python warnings.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": ["a"] * 30 + ["b"] * 30})
    y = np.array(([0, 1] * 15) + ([0, 1] * 15))
    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_out, _ = rs.get_resampled_data(df, y)

    res = rs.resample_result
    assert res["group_totals_after"] == {"a": 30, "b": 30}
    assert res["n_rows_out"] == 60 and len(x_out) == 60
    assert res["n_duplicated_rows"] == 0 and res["n_effective_rows"] == 60
    assert res["max_replication_factor"] == 1.0
    assert res["n_rows_without_a_recorded_group"] == 0
    assert set(res["cell_sizes_after"].values()) == {15}
    assert list(rs.fit_result.warnings) == []
    assert _messages(caught) == [], f"a balanced frame was caveated: {_messages(caught)}"


def test_resampler_says_so_when_no_row_carries_a_group_at_all():
    """Reachable only since missingness stopped being the group "nan": with all 40
    'gender' values missing there is no cell to balance. Measured after the fix:
    the frame comes back unchanged (40 rows), ``could_not_check`` reads "nothing was
    resampled: all 40 row(s) have a missing value in ['gender'] ...", and
    ``n_rows_without_a_recorded_group`` is 40. Before, this frame was balanced into
    one phantom group and reported as a completed resample.
    """
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"f": rng.normal(size=40), "gender": [np.nan] * 40})
    y = np.array([0, 1] * 20)
    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_out, _ = rs.get_resampled_data(df, y)

    assert len(x_out) == 40
    assert rs.resample_result["n_rows_without_a_recorded_group"] == 40
    assert rs.resample_result["max_replication_factor"] is None
    message = rs.resample_result["could_not_check"]
    assert "all 40 row(s) have a missing value" in message, message
    assert any("nothing was resampled" in m for m in _messages(caught))


# ===========================================================================
# A5  FairRepresentationTransformer.fit: the substituted feature values
# ===========================================================================


def test_fair_representation_counts_the_feature_values_its_fit_substituted():
    """Before (measured 2026-09-27) on 60 rows where 30 of the 'f1' values were
    missing: ``fit_result.warnings == []``, zero Python warnings, and
    ``_feature_mu`` [50.08643804, 0.13323775] with ``_feature_sd``
    [50.17814817, 1.00618775], where the mean of the 30 OBSERVED f1 values is
    100.17287608734327. ``Xnum = df[...].fillna(0.0)`` fed the standardisation
    statistics, so the "fit-time mean" every later transform imputes with was half
    invented, and the record was byte-identical in shape to a complete-data fit.

    After: ``_feature_mu[0]`` == 100.17287608734327 (the observed mean, nothing
    else), ``fit_metrics['n_feature_values_substituted']`` 30 with
    ``substituted_per_feature`` {'f1': 30, 'f2': 0}, and the substitution named on
    ``fit_result.warnings`` and as a Python warning.
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    gender = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    f1 = rng.normal(100.0, 5.0, 60)
    f1[:30] = np.nan
    df = pd.DataFrame({"f1": f1, "f2": rng.normal(0.0, 1.0, 60), "gender": gender})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=5
        ).fit(df)

    assert np.isnan(f1[:30]).all(), "the fixture no longer reaches the branch"
    observed_mean = float(np.nanmean(f1))
    assert observed_mean == pytest.approx(100.17287608734327, rel=1e-12)
    assert float(t._feature_mu[0]) == pytest.approx(observed_mean, rel=1e-12), (
        f"the fit-time mean is still built on the substituted zeros: {t._feature_mu}"
    )
    metrics = t.fit_result.fit_metrics
    assert metrics["n_feature_values_substituted"] == 30
    assert metrics["substituted_per_feature"] == {"f1": 30, "f2": 0}
    disclosed = _messages(caught) + list(t.fit_result.warnings)
    assert any("substituted" in m for m in disclosed), (
        f"30 invented feature values entered the encoder and its mean silently: {disclosed}"
    )


def test_control_fair_representation_keeps_the_real_mean_of_a_complete_frame():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on the same frame
    with NO missing value: ``n_feature_values_substituted`` 0,
    ``fit_result.warnings == []``, zero Python warnings, ``_feature_mu``
    [99.70624898545049, 0.13323774540595218] which is exactly the pair of column
    means, and ``transform`` returns a finite representation with
    ptp(rep_0) > 0.5 (2.06 on this run).
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    gender = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    df = pd.DataFrame(
        {"f1": rng.normal(100.0, 5.0, 60), "f2": rng.normal(0.0, 1.0, 60), "gender": gender}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=5
        ).fit(df)
        rep = t.transform(df)

    assert t.fit_result.fit_metrics["n_feature_values_substituted"] == 0
    assert float(t._feature_mu[0]) == pytest.approx(99.70624898545049, rel=1e-12)
    assert float(t._feature_mu[1]) == pytest.approx(0.13323774540595218, rel=1e-12)
    assert list(t.fit_result.warnings) == []
    assert _messages(caught) == [], f"a complete fit was caveated: {_messages(caught)}"
    assert bool(np.isfinite(rep.to_numpy()).all())
    assert float(np.ptp(rep["rep_0"].to_numpy())) > 0.5


def test_fair_representation_refuses_a_feature_with_no_observed_value():
    """A column with NO finite value has no observed mean to standardise or impute
    with, and filling it with 0.0 would put an invented value in the encoder AND in
    the statistics every later transform uses. Measured after the fix: ValueError
    naming ['f1']. Before, the fit succeeded with mu 0.0 and sd 1.0 for that column.
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "f1": np.full(60, np.nan),
            "f2": rng.normal(0.0, 1.0, 60),
            "gender": np.array(["a"] * 30 + ["b"] * 30, dtype=object),
        }
    )
    with pytest.raises(ValueError, match="no finite value"):
        FairRepresentationTransformer(
            protected_attributes=["gender"], representation_dim=2, epochs=3
        ).fit(df)


# ===========================================================================
# A6  ReweightingTransformer.fit: method='custom' counts the groupless rows
# ===========================================================================


def test_reweighting_custom_records_the_rows_that_have_no_group():
    """Before (measured 2026-09-27) on 60 rows with 10 missing 'race' values:
    method='custom' with target_distribution {'a': 0.5, 'b': 0.5} published
    ``n_rows_without_a_recorded_group`` 0 with ``fit_result.warnings == []`` and no
    Python warning, while method='inverse_frequency' on the SAME frame reported 10
    on both channels. ``n_rows_without_group`` was hardcoded to 0 for 'custom'
    because its uncovered groups have their own disclosure, but 'uncovered' comes
    from ``value_counts()``, which EXCLUDES NaN.

    After: both methods report 10 and carry the "belong to no known group" message
    on both channels, and the weights are unchanged (a 1.5, b 1.0).
    """
    rng = np.random.default_rng(1)
    race = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    race[:10] = np.nan
    df = pd.DataFrame({"f1": rng.normal(size=60), "race": race})
    y = (rng.random(60) < 0.5).astype(int)

    seen = {}
    for method, kwargs in (
        ("custom", {"target_distribution": {"a": 0.5, "b": 0.5}}),
        ("inverse_frequency", {}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = ReweightingTransformer(protected_attributes=["race"], method=method, **kwargs).fit(
                df, y
            )
        metrics = t.fit_result.fit_metrics
        disclosed = _messages(caught) + list(t.fit_result.warnings)
        seen[method] = (metrics["n_rows_without_a_recorded_group"], disclosed)

    assert seen["custom"][0] == 10, (
        f"the fit record says {seen['custom'][0]} rows without a group for a frame with 10"
    )
    assert seen["custom"][0] == seen["inverse_frequency"][0], (
        "two methods disagree about the same frame's groupless rows"
    )
    assert any("no known group" in m for m in seen["custom"][1]), seen["custom"][1]


def test_control_reweighting_custom_still_reports_its_measured_weights():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on 50 rows,
    'race' 20 'a' and 30 'b', NO missing value, target_distribution
    {'a': 0.5, 'b': 0.5}: ``n_rows_without_a_recorded_group`` 0, weights exactly
    a 1.25 and b 0.8333333333333334, ``get_sample_weights`` handing 1.25 to the
    first row and 0.8333333333333334 to the last, ``fit_result.warnings == []`` and
    no Python warning. And an UNCOVERED group still reports 0 groupless rows, so it
    is not double counted: it keeps its own "no entry for observed group(s)"
    disclosure.
    """
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"f1": rng.normal(size=50), "race": np.array(["a"] * 20 + ["b"] * 30, dtype=object)}
    )
    y = (rng.random(50) < 0.5).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(
            protected_attributes=["race"],
            method="custom",
            target_distribution={"a": 0.5, "b": 0.5},
        ).fit(df, y)
        w = t.get_sample_weights(df, y)

    metrics = t.fit_result.fit_metrics
    assert metrics["n_rows_without_a_recorded_group"] == 0
    assert float(metrics["group_weights"]["a"]) == pytest.approx(1.25, rel=1e-12)
    assert float(metrics["group_weights"]["b"]) == pytest.approx(0.8333333333333334, rel=1e-12)
    assert float(w[0]) == pytest.approx(1.25, rel=1e-12)
    assert float(w[-1]) == pytest.approx(0.8333333333333334, rel=1e-12)
    assert list(t.fit_result.warnings) == []
    assert _messages(caught) == [], f"a measured reweighting was caveated: {_messages(caught)}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        uncovered = ReweightingTransformer(
            protected_attributes=["race"], method="custom", target_distribution={"a": 1.0}
        ).fit(df, y)
    assert uncovered.fit_result.fit_metrics["n_rows_without_a_recorded_group"] == 0, (
        "an uncovered but PRESENT group was counted as a groupless row as well"
    )
    assert any("no entry for observed group" in m for m in _messages(caught))


# ===========================================================================
# A7  FeatureEngineeringAnalyzer.get_explanation: the features never screened
# ===========================================================================


def _categorical_proxy_frame(n: int = 300):
    rng = np.random.default_rng(4)
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    return pd.DataFrame(
        {
            "gender": gender,
            "zip_code": np.where(gender == "F", "48201", "90210"),
            "job_title": np.where(gender == "F", "nurse", "driver"),
            "hired": (rng.random(n) < 0.5).astype(int),
        }
    )


def test_an_analysis_that_screened_no_feature_at_all_withholds_its_verdict():
    """Before (measured 2026-09-27) on 300 rows where 'zip_code' is a PERFECT proxy
    for gender (F to '48201', M to '90210') and every feature is categorical:
    ``feature_columns == []``, ``screens_not_run == []``,
    ``ungraded_proxy_screens == []``, ``proxy_screen_complete`` True,
    ``risk_summary`` all zeros including ``not_assessed`` 0, zero warnings, and the
    explanation read "Analysed 0 features across 1 protected attribute(s). 0 proxy
    variable(s) and 0 high-risk feature(s) identified." at severity 'info', with
    'zip_code' named on no channel.

    The constructor kept only ``is_numeric_dtype`` columns as candidates and
    recorded the drop nowhere, so the report's own coverage lists (the qualifier's
    only discriminator) were empty because nothing was ever a candidate.

    After: ``screens_not_run`` names 'zip_code' and 'job_title' with the reason
    "never a candidate: the column is dtype object ...",
    ``proxy_screen_complete`` False, ``risk_summary['not_assessed']`` 2, severity
    'medium', the summary opens "COULD NOT CHECK: 2 screen(s) did not run", and
    full_analysis emits the UserWarning naming the count.
    """
    df = _categorical_proxy_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FeatureEngineeringAnalyzer(
            df, protected_attributes=["gender"], target_column="hired"
        )
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)

    assert analyzer.feature_columns == [], "the fixture no longer reaches the branch"
    named = {e["feature"] for e in report.screens_not_run}
    assert {"zip_code", "job_title"} <= named, (
        f"a feature dropped at construction is named nowhere: {report.screens_not_run}"
    )
    assert report.proxy_screen_complete is False
    assert report.risk_summary["not_assessed"] == 2
    assert "COULD NOT CHECK" in explanation.summary, (
        f"an analysis that screened nothing was summarised {explanation.summary[:120]!r} "
        f"at severity {explanation.severity!r}"
    )
    assert explanation.severity != "info"
    assert any("did not run" in m for m in _messages(caught))


def test_control_a_numeric_analysis_keeps_its_measured_verdict():
    """OVER-CORRECTION CONTROL, measured 2026-09-27 after the fix on 300 rows whose
    features are 'income' (40000 for F, 70000 for M) and 'noise': nothing is dropped
    at construction, ``screens_not_run == []``, ``proxy_screen_complete`` True,
    ``risk_summary`` {'critical': 1, ..., 'not_assessed': 0}, and the summary is
    the measured "Analysed 2 features across 1 protected attribute(s). 1 proxy
    variable(s) and 1 high-risk feature(s) identified." with no COULD NOT CHECK
    prefix. A fix that caveated every explanation would pass the pin above and
    destroy this.
    """
    rng = np.random.default_rng(4)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    df = pd.DataFrame(
        {
            "gender": gender,
            "income": np.where(gender == "F", 40000.0, 70000.0) + rng.normal(0, 1000, n),
            "noise": rng.normal(size=n),
            "hired": (rng.random(n) < 0.5).astype(int),
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FeatureEngineeringAnalyzer(
            df, protected_attributes=["gender"], target_column="hired"
        )
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)

    assert analyzer.feature_columns == ["income", "noise"]
    assert analyzer._features_excluded_at_construction == []
    assert report.screens_not_run == []
    assert report.proxy_screen_complete is True
    assert report.risk_summary["not_assessed"] == 0
    assert report.risk_summary["critical"] == 1
    assert "COULD NOT CHECK" not in explanation.summary
    assert explanation.summary.startswith("Analysed 2 features")
    assert not any("did not run" in m for m in _messages(caught))


def test_control_a_categorical_feature_named_explicitly_is_still_screened():
    """OVER-CORRECTION CONTROL for the other half: the drop is only recorded for the
    AUTO-detected feature set. Measured 2026-09-27 after the fix with
    feature_columns=['zip_code'] on the perfect-proxy frame:
    ``_features_excluded_at_construction == []``, ``screens_not_run == []``, and
    the screen grades 'zip_code' CRITICAL at correlation 0.9932. The fix must not
    turn a screened categorical column into a "never a candidate" entry.
    """
    df = _categorical_proxy_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FeatureEngineeringAnalyzer(
            df,
            protected_attributes=["gender"],
            target_column="hired",
            feature_columns=["zip_code"],
        )
        report = analyzer.full_analysis()

    assert analyzer._features_excluded_at_construction == []
    assert report.screens_not_run == []
    graded = {p.feature: p for p in report.proxy_variables}
    assert "zip_code" in graded, f"the named categorical proxy was not screened: {graded}"
    assert graded["zip_code"].risk_level.value == "critical"
    assert float(graded["zip_code"].correlation) == pytest.approx(0.9932, abs=5e-4)
    assert not any("never a candidate" in m for m in _messages(caught))
