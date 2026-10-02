"""Audit-6 lane 2 (feature engineering): a value that was never measured must not
be graded, plotted, weighted or reported as a measurement, and a documented option
that does nothing must be refused rather than relabelled.

Every finding below was reproduced by execution on 2026-09-09 before its fix:

  R-3  plot_transformation_comparison drew a green 0.000 "After" bar for an
       attribute absent from after_correlations (bar heights [0.8, 0], text
       '0.000', no warning); FeatureSuppressor.fit and ResidualTransformer.fit
       never populated correlation_after, so that was the chart every caller got.
  R-4  ReweightingTransformer.get_sample_weights handed a group unseen at fit the
       weight 1.0 silently ([0.75, 1.5, 1.0, 1.0] for A, B, C, C after fitting on
       A, B), and any unrecognised method fitted as a silent no-op (group_weights
       {}, every weight 1.0, fit_result.method 'reweighting_not_a_method').
  F3   ResidualTransformer accepted 'regression', 'quantile' and 'not_a_method',
       ran group_mean for all of them, and reported residualization_<name>.
  F11  CorrelationReducer stored target_correlation and never read it (outputs
       identical for 0.01 / 0.5 / 0.99); method='partial' was byte-identical to
       'residualize'.
  F15  compute_feature_correlations(include_pvalues=False) computed and returned
       p-values anyway.
  R-7  assess_geographic_feature_risk divided by ALL rows, so every ZIP the HOLC
       table could not resolve counted as grade A: 100 grade-D rows read 1.000
       'critical', the same 100 plus 900 unresolved read 0.100 'medium', and 50
       unresolved rows alone read 0.000 'low'.

Found while verifying R-3 by execution: _feature_attribute_association returned
eta between 0.5 and 1.0 for a CONSTANT column whose mean is not bit-exact under
pairwise summation (ss_total ~ 1e-26 of ulp noise), so a masked feature read as
a perfect proxy and the new after-measurement came out WORSE than before.

Each scenario carries an over-correction control asserting the healthy path
still gives its real answer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
    plt = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
    import matplotlib.pyplot as plt
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)

from vfairness.preprocessing.bias_detection.geographic_data import (
    assess_geographic_feature_risk,
)
from vfairness.preprocessing.feature_engineering import visualization as viz
from vfairness.preprocessing.feature_engineering.correlation import (
    compute_feature_correlations,
)
from vfairness.preprocessing.feature_engineering.transformers import (
    CorrelationReducer,
    FeatureSuppressor,
    ResidualTransformer,
    ReweightingTransformer,
    TransformationResult,
    _feature_attribute_association,
)

GRADE_D_ZIP = "48201"  # Detroit, grade D in KNOWN_REDLINED_ZIP_PATTERNS
GRADE_C_ZIP = "48209"  # Detroit, grade C
UNRESOLVED_ZIP = "99999"  # not in the table


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


def _eta_sq(feature: np.ndarray, groups: np.ndarray) -> float:
    """True correlation ratio eta^2 of a numeric feature vs a categorical attr."""
    x = np.asarray(feature, dtype=float)
    grand = x.mean()
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    g = np.asarray(groups)
    for level in pd.unique(g):
        xg = x[g == level]
        ss_between += len(xg) * (xg.mean() - grand) ** 2
    return ss_between / ss_total


def _nominal_frame(seed: int = 42, per_group: int = 300):
    """3-category nominal attribute with a NON-monotonic group-mean pattern
    (A and C high, B low) plus one independent feature."""
    rng = np.random.default_rng(seed)
    race = np.array((["A"] * per_group) + (["B"] * per_group) + (["C"] * per_group))
    feat = np.where(race == "B", 0.0, 10.0) + rng.normal(0.0, 1.0, len(race))
    indep = rng.normal(0.0, 1.0, len(race))
    return pd.DataFrame({"race": race, "feat": feat, "indep": indep}), race


def _messages(records) -> list[str]:
    return [str(r.message) for r in records]


# ========================================================================= #
# R-3: the before/after chart never draws a zero for a value nobody measured
# ========================================================================= #


class TestR3PlotterNeverDrawsAnUnmeasuredZero:
    @needs_matplotlib
    def test_absent_after_value_draws_no_bar_and_warns(self):
        """The exact reproduction: before={'race': 0.8}, after={} used to render
        heights [0.8, 0] annotated '0.800' / '0.000' with no warning."""
        with pytest.warns(
            UserWarning, match=r"no after-transformation correlation was measured for \['race'\]"
        ):
            ax = viz.plot_transformation_comparison({"race": 0.8}, {})
        heights = [b.get_height() for b in ax.patches]
        assert heights[0] == pytest.approx(0.8)
        assert np.isnan(heights[1]), (
            f"unmeasured 'After' slot must be NaN (no bar), got {heights[1]}"
        )
        texts = [t.get_text() for t in ax.texts]
        assert "not measured" in texts
        assert "0.000" not in texts, "a 0.000 annotation is the fabricated measurement"

    @needs_matplotlib
    def test_mixed_attributes_keep_measured_values_and_blank_the_rest(self):
        with pytest.warns(UserWarning, match=r"\['gender'\]"):
            ax = viz.plot_transformation_comparison({"race": 0.8, "gender": 0.4}, {"race": 0.1})
        heights = [b.get_height() for b in ax.patches]
        assert heights[:3] == pytest.approx([0.8, 0.4, 0.1])
        assert np.isnan(heights[3])
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["0.800", "0.400", "0.100", "not measured"]
        assert ax.get_ylim()[1] == pytest.approx(0.8 * 1.2)  # NaN excluded from the axis range

    @needs_matplotlib
    def test_a_measured_zero_is_still_a_bar_with_no_warning(self):
        """Over-correction control: a genuinely measured 0.0 keeps its bar."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = viz.plot_transformation_comparison({"race": 0.8}, {"race": 0.0})
        assert not [m for m in _messages(w) if "not measured" in m or "no after" in m]
        heights = [b.get_height() for b in ax.patches]
        assert heights == pytest.approx([0.8, 0.0])
        assert [t.get_text() for t in ax.texts] == ["0.800", "0.000"]

    @needs_matplotlib
    def test_effect_figure_labels_a_missing_after_panel(self):
        result = TransformationResult(
            method="m",
            n_features_original=2,
            n_features_transformed=2,
            n_samples=10,
            correlation_before={"race": 0.5},
        )
        fig = viz.plot_feature_transformation_effect(result)
        panel = fig.axes[0]
        assert not panel.patches, "no bars may be drawn when nothing was measured after"
        assert any("not measured" in t.get_text() for t in panel.texts)

    @needs_matplotlib
    def test_effect_figure_still_draws_bars_when_after_exists(self):
        result = TransformationResult(
            method="m",
            n_features_original=2,
            n_features_transformed=2,
            n_samples=10,
            correlation_before={"race": 0.5},
            correlation_after={"race": 0.1},
        )
        fig = viz.plot_feature_transformation_effect(result)
        panel = fig.axes[0]
        assert len(panel.patches) == 2
        assert [t.get_text() for t in panel.texts] == ["0.500", "0.100"]


class TestR3FitsMeasureCorrelationAfter:
    def test_feature_suppressor_remove_measures_after(self):
        df, _ = _nominal_frame()
        sup = FeatureSuppressor(
            protected_attributes=["race"], strategy="remove", correlation_threshold=0.3
        ).fit(df)
        res = sup.fit_result
        assert res is not None
        assert res.features_removed == ["feat"]
        assert res.correlation_before["race"] > 0.4
        assert "race" in res.correlation_after, "correlation_after was never populated (R-3)"
        assert res.correlation_after["race"] < 0.1
        assert res.correlation_reduction["race"] > 0.8
        assert res.warnings == []

    def test_feature_suppressor_mask_measures_after_as_no_association(self):
        df, _ = _nominal_frame()
        sup = FeatureSuppressor(
            protected_attributes=["race"], strategy="mask", correlation_threshold=0.3
        ).fit(df)
        res = sup.fit_result
        assert res is not None
        assert res.correlation_after["race"] < res.correlation_before["race"]
        assert res.correlation_after["race"] < 0.1

    def test_feature_suppressor_noise_stays_unmeasured_and_says_so(self):
        """'noise' redraws on every transform(), so a fit-time draw is not the
        output the caller gets: could-not-check, stated in the result."""
        df, _ = _nominal_frame()
        sup = FeatureSuppressor(
            protected_attributes=["race"], strategy="noise", correlation_threshold=0.3
        ).fit(df)
        res = sup.fit_result
        assert res is not None
        assert res.correlation_after == {}
        assert res.correlation_reduction == {}
        assert any("noise" in w and "not measured" in w for w in res.warnings)

    def test_residual_transformer_measures_after_for_the_conditioned_attribute(self):
        df, _ = _nominal_frame()
        r = ResidualTransformer(protected_attributes=["race"]).fit(df)
        res = r.fit_result
        assert res is not None
        assert res.correlation_before["race"] > 0.4
        assert "race" in res.correlation_after, "correlation_after was never populated (R-3)"
        assert res.correlation_after["race"] < 1e-6
        assert res.fit_metrics == {"conditioned_on": "race"}

    def test_residual_transformer_does_not_claim_removal_for_an_unconditioned_attribute(self):
        """Only the first attribute is conditioned on. The after-value for the
        second must show the association that remains. Its group pattern is
        non-monotonic across three categories, so Pearson on integer codes (the
        measure this class used until 2026-09-09) reads ~0 before AND after and
        would have reported a false success; eta sees it."""
        rng = np.random.default_rng(1)
        n = 900
        race = np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 300)
        attr2 = rng.choice(["X", "Y", "Z"], size=n)
        feat = (
            np.where(race == "B", 0.0, 10.0)
            + np.where(attr2 == "Y", 0.0, 10.0)
            + rng.normal(0.0, 1.0, n)
        )
        df = pd.DataFrame({"race": race, "attr2": attr2, "feat": feat})
        codes = pd.Categorical(attr2).codes.astype(float)
        assert abs(np.corrcoef(feat, codes)[0, 1]) < 0.05  # the old measure could not see it

        r = ResidualTransformer(protected_attributes=["race", "attr2"]).fit(df)
        res = r.fit_result
        assert res is not None
        assert res.correlation_before["attr2"] > 0.5
        assert res.correlation_after["race"] < 1e-6
        assert res.correlation_after["attr2"] > 0.5, (
            "the unconditioned attribute's association was reported as removed"
        )
        assert any("attr2" in w for w in res.warnings)

    def test_residual_transformer_warns_for_a_group_unseen_at_fit(self):
        df, _ = _nominal_frame()
        r = ResidualTransformer(protected_attributes=["race"]).fit(df)
        other = df.head(20).copy()
        other["race"] = "Z"
        with pytest.warns(UserWarning, match=r"not seen at fit \(\['Z'\]\)"):
            out = r.transform(other)
        # No conditional mean exists, so the rows pass through un-residualized.
        assert np.allclose(out["feat"].to_numpy(), other["feat"].to_numpy())

    def test_residual_transformer_seen_groups_transform_without_warning(self):
        df, race = _nominal_frame()
        r = ResidualTransformer(protected_attributes=["race"]).fit(df)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = r.transform(df)
        assert not [m for m in _messages(w) if "not seen at fit" in m]
        assert _eta_sq(out["feat"].to_numpy(), race) < 1e-6


class TestConstantColumnAssociation:
    """Found by execution while verifying R-3 (see module docstring)."""

    def test_constant_column_reads_zero_even_when_its_mean_is_not_bit_exact(self):
        race = pd.Series(np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 300))
        x = np.full(900, 13.279640262203964)
        assert x.mean() != x[0], "precondition: pairwise summation is one ulp off here"
        assert _feature_attribute_association(x, race) == 0.0

    def test_real_association_is_unchanged(self):
        df, _ = _nominal_frame()
        eta = _feature_attribute_association(df["feat"].to_numpy(), df["race"])
        assert eta is not None and eta > 0.95


# ========================================================================= #
# R-4: reweighting never hands out a weight it did not derive
# ========================================================================= #


def _train_frame():
    return pd.DataFrame({"grp": ["A"] * 40 + ["B"] * 20, "x": np.arange(60.0)})


class TestR4UnseenGroupWeightsAreNaN:
    def test_unseen_group_rows_are_nan_and_named(self):
        t = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        t.fit(_train_frame())
        other = pd.DataFrame({"grp": ["A", "B", "C", "C"], "x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.warns(
            UserWarning, match=r"no weight was fitted for group\(s\) of 'grp' \['C'\]; 2 row"
        ):
            w = t.get_sample_weights(other)
        assert w[0] == pytest.approx(0.75)
        assert w[1] == pytest.approx(1.5)
        assert np.isnan(w[2]) and np.isnan(w[3]), f"unseen rows must be NaN, got {w[2:]}"

    def test_seen_groups_keep_exact_weights_with_no_warning(self):
        train = _train_frame()
        t = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        t.fit(train)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            weights = t.get_sample_weights(train)
        assert not [m for m in _messages(w) if "no weight was fitted" in m]
        assert np.allclose(weights[:40], 0.75) and np.allclose(weights[40:], 1.5)
        assert weights.sum() == pytest.approx(60.0)

    def test_a_nan_weight_cannot_train_silently(self):
        """The reason NaN (not a raise) is the honest state for a training
        weight: the rows that WERE measured keep their weights, and the gap
        still cannot train unnoticed."""
        t = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        t.fit(_train_frame())
        other = pd.DataFrame({"grp": ["A"] * 30 + ["B"] * 29 + ["C"], "x": np.arange(60.0)})
        with pytest.warns(UserWarning):
            w = t.get_sample_weights(other)
        y = np.array([0, 1] * 30)
        with pytest.raises(ValueError, match="NaN"):
            LogisticRegression().fit(other[["x"]].to_numpy(), y, sample_weight=w)
        assert np.isnan(np.average(y, weights=w))

    def test_target_parity_cell_absent_at_fit_is_nan(self):
        grp = np.array(["A"] * 40 + ["B"] * 20)
        y = np.concatenate([np.repeat([1, 0], [25, 15]), np.zeros(20, dtype=int)])  # B: no y=1
        df = pd.DataFrame({"grp": grp, "x": np.arange(60.0)})
        t = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        t.fit(df, y)
        probe = pd.DataFrame({"grp": ["A", "B", "B"], "x": [1.0, 2.0, 3.0]})
        with pytest.warns(UserWarning, match=r"\(group, label\) cell"):
            w = t.get_sample_weights(probe, np.array([1, 0, 1]))
        assert np.isfinite(w[0]) and np.isfinite(w[1])
        assert np.isnan(w[2]), "the (B, 1) cell never existed at fit; its weight is unmeasured"


class TestR4UnknownMethodIsRefused:
    def test_constructor_refuses_an_unknown_method(self):
        with pytest.raises(ValueError, match="must be one of 'inverse_frequency'"):
            ReweightingTransformer(protected_attributes=["grp"], method="not_a_method")

    def test_fit_refuses_a_method_reassigned_after_construction(self):
        t = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        t.method = "not_a_method"
        with pytest.raises(ValueError, match="Unsupported reweighting method"):
            t.fit(_train_frame())
        assert not t.is_fitted

    def test_custom_with_no_matching_group_is_refused(self):
        t = ReweightingTransformer(
            protected_attributes=["grp"], method="custom", target_distribution={"X": 1.0}
        )
        with pytest.raises(ValueError, match="match none of the observed groups"):
            t.fit(_train_frame())

    def test_custom_partial_coverage_warns_and_uncovered_rows_are_nan(self):
        train = _train_frame()
        t = ReweightingTransformer(
            protected_attributes=["grp"], method="custom", target_distribution={"A": 1.0}
        )
        with pytest.warns(UserWarning, match=r"no entry for observed group\(s\) \['B'\]"):
            t.fit(train)
        with pytest.warns(UserWarning, match="no weight was fitted"):
            w = t.get_sample_weights(train)
        assert np.all(np.isfinite(w[:40]))
        assert np.all(np.isnan(w[40:]))

    def test_documented_methods_still_fit_and_weight(self):
        train = _train_frame()
        y = np.array([1, 0] * 30)
        inv = ReweightingTransformer(protected_attributes=["grp"], method="inverse_frequency")
        tp = ReweightingTransformer(protected_attributes=["grp"], method="target_parity")
        custom = ReweightingTransformer(
            protected_attributes=["grp"],
            method="custom",
            target_distribution={"A": 0.5, "B": 0.5},
        )
        for t, weights in (
            (inv, lambda: inv.fit(train).get_sample_weights(train)),
            (tp, lambda: tp.fit(train, y).get_sample_weights(train, y)),
            (custom, lambda: custom.fit(train).get_sample_weights(train)),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = weights()
            assert not [m for m in _messages(w) if "no weight was fitted" in m], t.method
            assert np.all(np.isfinite(out)), t.method
            assert out.sum() == pytest.approx(60.0), t.method


# ========================================================================= #
# F3: ResidualTransformer refuses what it does not implement
# ========================================================================= #


class TestF3ResidualTransformerRefusesUnimplementedMethods:
    @pytest.mark.parametrize("method", ["regression", "quantile", "not_a_method"])
    def test_refused_at_construction(self, method):
        with pytest.raises(NotImplementedError, match="only 'group_mean'"):
            ResidualTransformer(protected_attributes=["race"], method=method)

    def test_group_mean_still_works_and_is_labelled_truthfully(self):
        df, race = _nominal_frame()
        r = ResidualTransformer(protected_attributes=["race"], method="group_mean")
        out = r.fit_transform(df)
        assert r.fit_result is not None
        assert r.fit_result.method == "residualization_group_mean"
        assert _eta_sq(out["feat"].to_numpy(), race) < 1e-6


# ========================================================================= #
# F11: CorrelationReducer: no dead target, no alias method
# ========================================================================= #


class TestF11CorrelationReducerTargetAndPartial:
    def test_partial_is_refused(self):
        with pytest.raises(NotImplementedError, match="method='partial'"):
            CorrelationReducer(protected_attributes=["race"], method="partial")

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="must be 'residualize' or 'decorrelate'"):
            CorrelationReducer(protected_attributes=["race"], method="orthogonalise")

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_target_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match=r"target_correlation must be in \[0, 1\]"):
            CorrelationReducer(protected_attributes=["race"], target_correlation=bad)

    def test_target_is_verified_against_the_measurement_and_reported_met(self):
        df, _ = _nominal_frame()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = CorrelationReducer(protected_attributes=["race"], target_correlation=0.1).fit(df)
        assert not [m for m in _messages(w) if "exceeds target_correlation" in m]
        res = r.fit_result
        assert res is not None
        assert res.fit_metrics["target_correlation"] == 0.1
        assert res.fit_metrics["target_met"] == {"race": True}
        assert res.correlation_after["race"] <= 0.1

    def test_unmet_target_is_reported_false_and_warned(self):
        """The verification logic on its own: force the after-measurement above
        the target and the result must say so instead of claiming success."""
        df, _ = _nominal_frame()
        r = CorrelationReducer(protected_attributes=["race"], target_correlation=0.1)
        r._compute_correlations = lambda frame, cols, attrs: {"race": 0.42}  # type: ignore[method-assign]
        with pytest.warns(UserWarning, match="exceeds target_correlation=0.1"):
            r.fit(df)
        assert r.fit_result is not None
        assert r.fit_result.fit_metrics["target_met"] == {"race": False}

    def test_unmeasurable_attribute_reports_none_not_met(self):
        """No numeric feature has 10 valid rows: nothing can be measured, so the
        attribute is omitted from the report (not 0.0) and the target verdict is
        None, not True."""
        race = np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 300)
        feat = np.full(900, np.nan)
        feat[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"race": race, "feat": feat})
        with pytest.warns(UserWarning, match="no feature/attribute association could be measured"):
            r = CorrelationReducer(protected_attributes=["race"]).fit(df)
        res = r.fit_result
        assert res is not None
        assert res.correlation_before == {} and res.correlation_after == {}
        assert res.fit_metrics["target_met"] == {"race": None}
        assert res.correlation_reduction == {}

    def test_output_does_not_depend_on_the_target(self):
        """The documented claim, pinned: the target is verified, not applied."""
        df, _ = _nominal_frame()
        low = CorrelationReducer(protected_attributes=["race"], target_correlation=0.05)
        high = CorrelationReducer(protected_attributes=["race"], target_correlation=0.6)
        assert np.allclose(low.fit_transform(df).to_numpy(), high.fit_transform(df).to_numpy())

    @pytest.mark.parametrize("method", ["residualize", "decorrelate"])
    def test_documented_methods_still_remove_the_association(self, method):
        df, race = _nominal_frame()
        reducer = CorrelationReducer(
            protected_attributes=["race"], method=method, preserve_variance=False
        )
        out = reducer.fit_transform(df)
        assert _eta_sq(out["feat"].to_numpy(), race) < 1e-6
        assert reducer.fit_result is not None
        assert reducer.fit_result.method == f"correlation_reduction_{method}"


# ========================================================================= #
# F15: compute_feature_correlations has no switch that does nothing
# ========================================================================= #


class TestF15IncludePvaluesIsGone:
    def test_the_dead_kwarg_is_refused(self):
        df, _ = _nominal_frame()
        with pytest.raises(TypeError, match="include_pvalues"):
            compute_feature_correlations(df, protected_attributes=["race"], include_pvalues=False)

    def test_pvalues_are_always_returned(self):
        df, _ = _nominal_frame()
        m = compute_feature_correlations(df, protected_attributes=["race"])
        assert np.isfinite(m.pvalues.loc["feat", "race"])
        assert m.pvalues.loc["feat", "race"] < 0.01
        assert abs(m.correlations.loc["feat", "race"]) > 0.9

    def test_unknown_method_is_refused(self):
        df, _ = _nominal_frame()
        with pytest.raises(ValueError, match="method must be one of"):
            compute_feature_correlations(df, protected_attributes=["race"], method="bogus")

    @pytest.mark.parametrize("method", ["auto", "pearson", "spearman", "cramers_v"])
    def test_documented_methods_still_work(self, method):
        df, _ = _nominal_frame()
        m = compute_feature_correlations(df, protected_attributes=["race"], method=method)
        assert list(m.correlations.index) == ["feat", "indep"]
        assert m.method == method


# ========================================================================= #
# R-7: an unresolvable ZIP is could-not-check, not HOLC grade A
# ========================================================================= #


class TestR7UnresolvedZipsAreNotGradeA:
    def test_unresolved_rows_do_not_dilute_the_score(self):
        """The exact reproduction: 100 grade-D rows plus 900 unresolved used to
        read 0.100 'medium'."""
        a = assess_geographic_feature_risk([GRADE_D_ZIP] * 100 + [UNRESOLVED_ZIP] * 900)
        assert a.risk_score == pytest.approx(1.0)
        assert a.holc_coverage == pytest.approx(0.1)
        assert a.disparate_impact_risk.startswith("critical"), a.disparate_impact_risk
        assert "10.0% of rows with HOLC data" in a.disparate_impact_risk
        assert "900 of 1000 unresolved" in a.disparate_impact_risk
        assert "Consider removing or aggregating this geographic feature" in a.recommendations
        assert any("rests on 100 resolved row(s) out of 1000" in r for r in a.recommendations)

    def test_fully_resolved_verdict_is_unqualified(self):
        d = assess_geographic_feature_risk([GRADE_D_ZIP] * 100)
        assert d.risk_score == pytest.approx(1.0)
        assert d.disparate_impact_risk == "critical"
        c = assess_geographic_feature_risk([GRADE_C_ZIP] * 100)
        assert c.risk_score == pytest.approx(0.7)
        assert c.disparate_impact_risk == "critical"
        assert c.holc_coverage == pytest.approx(1.0)

    def test_mixed_grades_score_over_resolved_rows_only(self):
        a = assess_geographic_feature_risk(
            [GRADE_D_ZIP] * 50 + [GRADE_C_ZIP] * 50 + [UNRESOLVED_ZIP] * 100
        )
        assert a.risk_score == pytest.approx((50 * 1.0 + 50 * 0.7) / 100)
        assert a.affected_samples == 100
        # Coverage of exactly one half is not a minority: no qualifier.
        assert a.holc_coverage == pytest.approx(0.5)
        assert a.disparate_impact_risk == "critical"

    def test_majority_coverage_verdict_is_unqualified(self):
        a = assess_geographic_feature_risk([GRADE_D_ZIP] * 100 + [UNRESOLVED_ZIP] * 50)
        assert a.risk_score == pytest.approx(1.0)
        assert a.disparate_impact_risk == "critical"
        assert not any("Only" in r and "HOLC data" in r for r in a.recommendations)

    def test_no_resolved_rows_is_unknown_not_low(self):
        """50 unresolved rows alone used to read 0.000 'low'."""
        with pytest.warns(UserWarning, match="none of the 50 value"):
            a = assess_geographic_feature_risk([UNRESOLVED_ZIP] * 50)
        assert np.isnan(a.risk_score)
        assert a.disparate_impact_risk == "unknown"
        assert a.holc_coverage == 0.0
        assert a.affected_samples == 0

    def test_empty_input_is_nan_unknown(self):
        with pytest.warns(UserWarning, match="no geographic values"):
            a = assess_geographic_feature_risk([])
        assert np.isnan(a.risk_score)
        assert a.disparate_impact_risk == "unknown"

    def test_unsupported_feature_type_is_nan_unknown(self):
        with pytest.warns(UserWarning, match="'county' is not supported"):
            a = assess_geographic_feature_risk([GRADE_D_ZIP], feature_type="county")
        assert np.isnan(a.risk_score)
        assert a.disparate_impact_risk == "unknown"
