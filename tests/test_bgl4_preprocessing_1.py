"""BGL4 AUDIT of batch A-preprocessing-1: the overturns, as executable evidence.

CLOSED 2026-09-27 (BGL5). Written by an AUDITOR against the unfixed library, where
every assertion below failed and the tests carried ``xfail(strict=True)``. All nine
defects are now fixed in ``preprocessing/feature_engineering/transformers.py`` and
``preprocessing/feature_engineering/analyzer.py``, so the marks are GONE and each
test asserts the corrected behaviour directly: the subjects, the fixtures and the
measured before-values in the docstrings are unchanged, because they are the record
of what the defect was. The exact after-values, the over-correction controls and
the sabotage log live in ``tests/test_bgl5_preprocessing_1.py``.

Each test below demonstrates one grade in
``/tmp/claude-501/bgl/audit_batches2/A-preprocessing-1.json``
that did NOT survive being attacked, and every number in a docstring was printed
by running the unit on this repo on 2026-09-27.

Every one of the eleven recorded sabotages was re-run first, against a modified
COPY of the module (never the shared checkout), and every one of them reddened its
pin. So none of these overturns is "the pin does not work". All of them are the
other failure the brief names: a SECOND unmeasurable input the pin never tried,
and in four cases a guard that structurally CANNOT fire for it.

THE COMMON ROOT of six of the seven: this file's own fixes taught four classes
that ``astype(str)`` turns a MISSING protected value into the level "nan" and that
a row whose group was never fitted must be counted, not corrected. The wave fixed
the classes it probed. The same two inputs were never tried on the three that
share the module with them:

    Resampler.get_resampled_data          still does ``astype(str)``, so "nan" is
                                          a group, is oversampled, and makes the
                                          one-group refusal unreachable
    ResidualTransformer.fit / .transform   records ``nan`` in ``_fitted_groups``,
                                          so its unseen-group refusal cannot fire
                                          for the rows that most need it
    FairRepresentationTransformer.fit      substitutes 0.0 for a missing FEATURE
                                          value, uncounted, and the fit-time mean
                                          every later transform uses is built on it
    LabelMassager.get_massaged_labels      the group-count guard reads the FIT, so a
                                          call-time frame with neither fitted group
                                          returns "0 flips" with nothing said
    FeatureSuppressor.fit / .transform     the nan-mask refusal sits under
                                          ``elif self.strategy == "mask"``, so
                                          strategy='noise' keeps the old behaviour
    ReweightingTransformer.fit             method='custom' hardcodes
                                          n_rows_without_a_recorded_group to 0
    FeatureEngineeringAnalyzer.get_explanation
                                          reads only the report's two coverage
                                          lists, and a feature dropped at
                                          CONSTRUCTION is in neither

The marks WERE ``xfail(strict=True)`` (``xfail_strict`` is on for this repo), so the
day these were fixed the strict xfails became FAILURES and the findings could not be
closed silently. That is what happened, and the marks were removed rather than the
tests: each one is now a pin on the repaired behaviour.
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
# A1  LabelMassager.get_massaged_labels, graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: get_massaged_labels now counts the deprived and the favored
# rows of THIS frame and warns with both counts. Pin and control:
# tests/test_bgl5_preprocessing_1.py::test_label_massager_names_a_call_frame_that_carries_no_comparable_group
@pytest.mark.parametrize(
    ("label", "level"),
    [
        ("a level the fit never saw", "c"),
        ("only the favored group", "b"),
        ("no group at all", None),
    ],
)
def test_label_massager_says_so_when_the_frame_carries_no_comparable_group(label, level):
    """Measured 2026-09-27. Fitted on the 200-row two-group frame (n_groups 2,
    ranker 'logistic', deprived 'a', favored 'b', 60 labels flipped at fit), then
    called on 40 rows of gender 'c' / all 'b' / all missing: the array came back
    byte-identical to the input, ``n_labels_flipped_`` 0 and ``warnings == []`` in
    all three cases.

    ``_planned_flip_count`` returns 0 when ``n_dep == 0 or n_fav == 0``, and the
    only guards in ``get_massaged_labels`` are ``self._model is None`` and
    ``self._n_groups_fitted < 2``, both properties of the FIT. This is the exact
    surface P1-5 was charged for (zero flips is not zero disparity), reached
    through the call-time frame instead of the fitted one, and the two siblings
    that condition on a fitted group (CorrelationReducer.transform,
    ResidualTransformer.transform) both disclose this state.
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
    assert any("could-not-check" in m for m in _messages(caught)), (
        f"0 flips over a comparison that never happened reached the caller silently: "
        f"{_messages(caught)}"
    )


# ===========================================================================
# A2  ResidualTransformer.fit and .transform, both graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: _fitted_groups is now ['a', 'b'] (a missing value is not a
# fitted group), and both fit() and transform() count those rows. Pin and control:
# tests/test_bgl5_preprocessing_1.py::test_residual_transformer_counts_the_rows_it_could_not_residualize
def test_residual_transformer_counts_the_rows_it_could_not_residualize():
    """Measured 2026-09-27 on 60 rows whose 'gender' held two real levels plus 10
    missing values: ``_group_means['f']`` keys ['a', 'b'] (missingness is correctly
    NOT a group), ``_fitted_groups`` [nan, 'a', 'b'], ``features_modified`` ['f'],
    ``fit_result.warnings == []``, zero Python warnings, and the 10 rows with no
    group came back byte-identical to their input while the other 50 moved.

    ``transform`` computes ``unseen = [g for g in pd.unique(df[attr]) if g not in
    self._fitted_groups]``, and nan IS in ``_fitted_groups`` (``list(pd.unique(...))``
    at fit keeps it), so the guard is unreachable for exactly the rows that have no
    conditional mean. The sibling DisparateImpactRemover publishes
    ``n_rows_without_a_recorded_group`` and warns that those rows are returned
    UNREPAIRED; this class says nothing on either channel.
    """
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rt = ResidualTransformer(protected_attributes=["gender"]).fit(df)
        out = rt.transform(df)

    raw, got = df["f"].to_numpy(), out["f"].to_numpy()
    assert np.allclose(got[:10], raw[:10]), "the fixture no longer reaches the branch"
    assert not np.allclose(got[10:], raw[10:]), "nothing was residualized at all"
    disclosed = _messages(caught) + list(rt.fit_result.warnings)
    assert any("no known group" in m or "no conditional mean" in m for m in disclosed), (
        f"10 of 60 rows were returned un-residualized inside the residual column with "
        f"nothing said: {disclosed}"
    )


# ===========================================================================
# A3  FeatureSuppressor.fit and .transform, both graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: a noise scale that perturbs nothing is refused at fit and joins
# features_not_suppressed, and transform applies the FITTED scale or nothing. Pin and
# control: tests/test_bgl5_preprocessing_1.py::test_feature_suppressor_noise_refuses_a_scale_that_perturbs_nothing
@pytest.mark.parametrize(
    ("label", "column"),
    [("a constant column", np.full(60, 7.0)), ("a column with no value", np.full(60, np.nan))],
)
def test_feature_suppressor_noise_does_not_report_a_suppression_that_did_nothing(label, column):
    """Measured 2026-09-27 with strategy='noise' and features_to_suppress=['proxy']:
    for a CONSTANT column ``nanstd`` is 0.0, so the noise drawn is exactly 0.0 and
    the output is byte-identical to the input; for an all-NaN column ``nanstd`` is
    nan and the column comes back untouched. In both cases
    ``features_modified == ['proxy']``, ``features_not_suppressed == []`` and no
    channel says the suppression did not happen (the all-NaN case emits only
    numpy's own "Degrees of freedom <= 0 for slice").

    The fit already owns the right disclosure: a non-finite mask value joins
    ``features_not_suppressed``. That branch is guarded by
    ``elif self.strategy == "mask"``, and 'bin' has its own try/except, so 'noise'
    is the one strategy with no degeneracy check at all. This is P1-6 and P1-8 for
    the third strategy.
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
    disclosed = _messages(caught) + list(fs.fit_result.warnings)
    assert fs.fit_result.features_modified == [] or any(
        "could not be suppressed" in m for m in disclosed
    ), (
        f"a suppression that changed nothing was published as a modification: "
        f"features_modified={fs.fit_result.features_modified}, disclosed={disclosed}"
    )


# ===========================================================================
# A4  Resampler.get_resampled_data, graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: the cell key is masked like its three siblings', the groupless
# rows are excluded from the balancing, counted and returned unbalanced, and the
# single-group refusal fires. Pin and control:
# tests/test_bgl5_preprocessing_1.py::test_resampler_does_not_balance_against_a_group_made_of_missing_values
def test_resampler_does_not_balance_against_a_group_made_of_missing_values():
    """Measured 2026-09-27 on 60 rows whose 'gender' held ONE observed level 'a'
    plus 10 missing values: ``cell_sizes_before`` {'(nan, label=1)': 10,
    '(a, label=1)': 20, '(a, label=0)': 30}, ``cell_sizes_after`` all three at 30,
    ``group_totals_after`` {'nan': 30, 'a': 60}, and the report reasons about the
    phantom group in prose ("(nan, label=0)" listed as a cell with no examples).
    ``could_not_check`` was None.

    ``observed_group_keys`` is derived from the same astype(str) values, so
    ``len(observed_group_keys) < 2`` is False and the disclosure "between-group
    balance was NOT measured: only one protected group appears in these rows"
    cannot fire on the one frame it was written for. 20 of the 30 rows in that cell
    are copies of rows whose protected group is unknown. The three siblings that
    key on a group (ReweightingTransformer, LabelMassager, DisparateImpactRemover)
    all mask the key; this one does not.
    """
    rng = np.random.default_rng(0)
    gender = np.array(["a"] * 60, dtype=object)
    gender[:10] = np.nan
    df = pd.DataFrame({"f": rng.normal(size=60), "gender": gender})
    y = np.r_[np.ones(30, int), np.zeros(30, int)]

    rs = Resampler(protected_attributes=["gender"]).fit(df, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rs.get_resampled_data(df, y)

    keys = list(rs.resample_result["cell_sizes_before"]) + list(
        rs.resample_result["group_totals_after"]
    )
    disclosed = _messages(caught) + list(rs.fit_result.warnings)
    assert not any("nan" in str(k) for k in keys), (
        f"missingness was coined into a resampled group: {keys}"
    )
    assert any("only one protected group" in m for m in disclosed), (
        f"a single-group frame was resampled as if it had two: {disclosed}"
    )


# ===========================================================================
# A5  FairRepresentationTransformer.fit, graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: the standardisation statistics are taken over the OBSERVED
# values (mu[0] 100.17287608734327, not 50.086438), the substitution is filled with
# that observed mean and counted on both channels. Pin and control:
# tests/test_bgl5_preprocessing_1.py::test_fair_representation_counts_the_feature_values_its_fit_substituted
def test_fair_representation_counts_the_feature_values_its_fit_substituted():
    """Measured 2026-09-27 on 60 rows where 30 of the 'f1' values were missing:
    ``fit_result.warnings == []``, zero Python warnings, ``fit_metrics``
    byte-identical in shape to a complete-data fit, and ``_feature_mu[0]``
    50.086438 where the mean of the 30 OBSERVED values is 100.172876. The stored
    standardisation statistics are therefore half invented, and every later
    ``transform`` imputes with them while ``transform_result`` labels the
    substitution "fit-time mean".

    This class's own ``transform`` already counts substitutions
    (``n_values_imputed``, ``imputed_per_feature``, plus a UserWarning), and the
    sibling LabelMassager.fit publishes ``n_features_imputed`` on ``fit_metrics``
    AND in ``fit_result.warnings`` for exactly this input (that disclosure is
    itself a PROVEN row in this batch). Only this fit is silent.
    """
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
    disclosed = _messages(caught) + list(t.fit_result.warnings)
    assert any(
        "imput" in m.lower() or "substitut" in m.lower() or "filled" in m.lower() for m in disclosed
    ), f"30 invented feature values entered the encoder and its mean silently: {disclosed}"


# ===========================================================================
# A6  ReweightingTransformer.fit, graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: the groupless rows are counted directly from the key each
# method groups by, so 'custom' reports 10 like its sibling. Pin and control:
# tests/test_bgl5_preprocessing_1.py::test_reweighting_custom_records_the_rows_that_have_no_group
def test_reweighting_custom_records_the_rows_that_have_no_group():
    """Measured 2026-09-27 on 60 rows with 10 missing 'race' values. With
    method='inverse_frequency' the fit reports
    ``n_rows_without_a_recorded_group`` 10 and carries the "belong to no known
    group" message on ``fit_result.warnings`` AND as a Python warning. With
    method='custom' and target_distribution {'a': 0.5, 'b': 0.5} the same frame
    reports 0 and ``fit_result.warnings == []``.

    The zero is deliberate ("'custom' is left out because its uncovered groups
    already have their own disclosure"), but ``uncovered`` is computed from
    ``df[attr].value_counts()``, which EXCLUDES NaN, so a missing protected value
    gets no fit-side disclosure of any kind. ``get_sample_weights`` does hand those
    rows NaN with a warning, so the array surface is honest and only the serialised
    record is wrong, which is why this is recorded as a false count rather than a
    fabricated weight.
    """
    rng = np.random.default_rng(1)
    race = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    race[:10] = np.nan
    df = pd.DataFrame({"f1": rng.normal(size=60), "race": race})
    y = (rng.random(60) < 0.5).astype(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = ReweightingTransformer(
            protected_attributes=["race"],
            method="custom",
            target_distribution={"a": 0.5, "b": 0.5},
        ).fit(df, y)

    metrics = t.fit_result.fit_metrics
    disclosed = _messages(caught) + list(t.fit_result.warnings)
    assert metrics["n_rows_without_a_recorded_group"] == 10 or any(
        "no known group" in m for m in disclosed
    ), (
        f"the fit record says {metrics['n_rows_without_a_recorded_group']} rows without a "
        f"group for a frame with 10, and warnings are {disclosed}"
    )


# ===========================================================================
# A7  FeatureEngineeringAnalyzer.get_explanation, graded PROVEN
# ===========================================================================


# CLOSED 2026-09-27: the columns the numeric-only auto-detection drops are recorded
# and reported as screens that did not run, so the qualifier, the not_assessed band
# and the UserWarning all see them. Pin and controls:
# tests/test_bgl5_preprocessing_1.py::test_an_analysis_that_screened_no_feature_at_all_withholds_its_verdict
def test_an_analysis_that_screened_no_feature_at_all_withholds_its_verdict():
    """Measured 2026-09-27 on 300 rows where 'zip_code' is a PERFECT
    proxy for gender (F to '48201', M to '90210') and every feature is categorical:
    ``feature_columns == []``, ``screens_not_run == []``,
    ``ungraded_proxy_screens == []``, ``proxy_screen_complete`` True,
    ``risk_summary`` all zeros including ``not_assessed`` 0, zero warnings, and the
    explanation reads "Analysed 0 features across 1 protected attribute(s). 0 proxy
    variable(s) and 0 high-risk feature(s) identified." at severity 'info'.
    'zip_code' is never named on any channel.

    This is the defect the row says was fixed, reached through a different input:
    the constructor keeps only ``is_numeric_dtype`` columns as candidates and
    records the drop nowhere, so the report's own coverage lists (the qualifier's
    only discriminator) are empty because nothing was ever a candidate rather than
    because everything was screened.
    """
    rng = np.random.default_rng(4)
    n = 300
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    df = pd.DataFrame(
        {
            "gender": gender,
            "zip_code": np.where(gender == "F", "48201", "90210"),
            "job_title": np.where(gender == "F", "nurse", "driver"),
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

    assert analyzer.feature_columns == [], "the fixture no longer reaches the branch"
    surfaced = _messages(caught) + [explanation.summary, report.summary]
    assert "COULD NOT CHECK" in explanation.summary or any("zip_code" in s for s in surfaced), (
        f"an analysis that screened nothing was summarised {explanation.summary[:120]!r} "
        f"at severity {explanation.severity!r}"
    )
