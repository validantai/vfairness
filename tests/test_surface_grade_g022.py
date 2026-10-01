"""Grading batch g022: the public surface of ``feature_engineering.analyzer``.

Every test here asserts at a public entry point of
:class:`vfairness.preprocessing.feature_engineering.FeatureEngineeringAnalyzer`,
and every could-not-check assertion has a CONTROL beside it computing the
expected value independently with ``numpy``. A fix that makes everything refuse
is a worse defect than the one it replaces and passes any test that only checks
the degenerate case.

Three fabrications were found by execution and are fixed in
``analyzer.py``; each has its section below.

1. ``compare_transformations`` reported ``avg_correlation = 0.0`` for a frame on
   which not one correlation could be computed. 0.0 is the BEST value on that
   scale (zero association with the protected attribute) and sorts FIRST, so
   ranking the table put the fabrication at the top.
2. ``analyze_proxies`` graded an unmeasurable (NaN) correlation NEGLIGIBLE, the
   safest band, and stamped ``evidence['pvalue_status'] = 'computed'`` over a
   NaN p-value, while reporting ``complete = True``.
3. ``get_feature_recommendations`` returned ``[]`` over a frame the screen never
   ran on, which reads as "nothing to do".

A fourth, smaller one: ``analyze_proxies(correlation_threshold=0.0)`` silently
used the constructor default, because the test was ``or`` rather than
``is not None``. Asking to see every pair therefore returned fewer.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.feature_engineering.analyzer import (
    FeatureEngineeringAnalyzer,
)


def _messages(records: List[Any]) -> str:
    return " || ".join(str(r.message) for r in records)


def _run(fn, *args, **kwargs) -> Tuple[Any, str]:
    """Call ``fn`` with warnings ENABLED and return (result, joined messages).

    A refusal carried only in a warning is invisible if warnings are suppressed,
    so every call in this file goes through here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, _messages(caught)


def _proxy_frame(n: int, *, constant_attr: bool = False, seed: int = 0) -> pd.DataFrame:
    """race, a near-perfect proxy for it (income), and an unrelated column."""
    rng = np.random.default_rng(seed)
    race = np.zeros(n, int) if constant_attr else (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame(
        {
            "race": race,
            "income": race * 50000 + rng.normal(0, 500, n),
            "noise": rng.normal(size=n),
        }
    )


def _expected_abs_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """|Pearson r| computed here, independently of the library."""
    return float(abs(np.corrcoef(df[a].to_numpy(float), df[b].to_numpy(float))[0, 1]))


# ---------------------------------------------------------------------------
# 1. compare_transformations: 0.0 is the best score on this scale
# ---------------------------------------------------------------------------


class TestCompareTransformationsRefusesInsteadOfScoringZero:
    """Measured before the fix, on a frame whose feature columns were all NaN::

                          method  corr_race  avg_correlation
        0               original        0.0              0.0
        1  correlation_reduction        0.0              0.0
        2    feature_suppression        0.0              0.0
        3        residualization        0.0              0.0

    Four perfect scores over a frame holding no measurable number anywhere, and
    the ``original`` row carried no warning of its own. Five rows and zero rows
    produced the same table.
    """

    def test_no_measurable_pair_is_nan_not_zero(self):
        df = _proxy_frame(400)
        df["income"] = np.nan
        df["noise"] = np.nan

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        table, messages = _run(analyzer.compare_transformations, methods=["residualization"])

        original = table[table["method"] == "original"].iloc[0]
        assert math.isnan(original["avg_correlation"])
        assert math.isnan(original["corr_race"])
        assert original["status"] == "not_measured"
        assert original["pairs_measured"] == 0
        assert original["pairs_not_measured"] == 2
        # The refusal has to be legible without reading the source.
        assert "NOT 0.0" in messages
        # 0.0 is the minimum of this scale, so it would rank first.
        assert not (table["avg_correlation"] == 0.0).any()

    def test_too_few_rows_is_nan_not_zero(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(5), ["race"], feature_columns=["income", "noise"]
        )
        table, messages = _run(analyzer.compare_transformations, methods=["residualization"])

        assert table["status"].tolist() == ["not_measured", "not_measured"]
        assert table["avg_correlation"].isna().all()
        assert "NOT 0.0" in messages

    def test_a_method_that_raises_gets_a_row_saying_so(self):
        """It used to vanish from the table, so a reader comparing three
        methods saw two rows and could not tell a method that blew up from one
        never requested."""
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(400), ["race"], feature_columns=["income", "noise"]
        )
        table, messages = _run(analyzer.compare_transformations, methods=["no_such_method"])

        assert table["method"].tolist() == ["original", "no_such_method"]
        failed = table.iloc[1]
        assert str(failed["status"]).startswith("failed: ValueError")
        assert math.isnan(failed["avg_correlation"])
        assert "raised ValueError" in messages

    def test_control_healthy_data_still_measures_the_real_correlation(self):
        """OVER-CORRECTION CONTROL. The expected value is computed here with
        numpy, not copied from what the code returns."""
        df = _proxy_frame(400)
        expected = float(
            np.mean(
                [
                    _expected_abs_corr(df, "income", "race"),
                    _expected_abs_corr(df, "noise", "race"),
                ]
            )
        )

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        table, messages = _run(analyzer.compare_transformations, methods=["residualization"])

        original = table[table["method"] == "original"].iloc[0]
        assert original["corr_race"] == pytest.approx(expected, rel=1e-9)
        assert original["avg_correlation"] == pytest.approx(expected, rel=1e-9)
        assert original["status"] == "measured"
        assert original["pairs_measured"] == 2
        assert original["pairs_not_measured"] == 0
        assert "NOT 0.0" not in messages
        # And the transformation still demonstrably reduces the association.
        assert table.iloc[1]["avg_correlation"] < original["avg_correlation"]

    def test_control_one_unmeasurable_column_does_not_discard_the_other(self):
        """THE REVERSE DEFECT. One constant column among two used to turn the
        whole attribute NaN, throwing away a real measured 0.9998. The mean is
        taken over the pairs that WERE measurable, and the counts disclose it."""
        df = _proxy_frame(400)
        df["noise"] = 1.0
        expected = _expected_abs_corr(df, "income", "race")

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        table, _ = _run(analyzer.compare_transformations, methods=["residualization"])

        original = table[table["method"] == "original"].iloc[0]
        assert original["corr_race"] == pytest.approx(expected, rel=1e-9)
        assert original["status"] == "partial"
        assert original["pairs_measured"] == 1
        assert original["pairs_not_measured"] == 1


# ---------------------------------------------------------------------------
# 2. analyze_proxies: the sibling of the fix full_analysis already had
# ---------------------------------------------------------------------------


class TestAnalyzeProxiesSaysWhenItCouldNotGrade:
    def test_a_nan_correlation_is_not_graded_negligible(self):
        """Measured before the fix on a constant protected attribute over 500
        rows: two ProxyVariableResult rows, ``risk_level=NEGLIGIBLE``,
        ``correlation=nan``, ``pvalue=nan``, ``pvalue_status='computed'`` and
        ``complete=True``. NEGLIGIBLE's action is "Minimal risk. No action
        needed."."""
        df = _proxy_frame(500, constant_attr=True)
        assert df["race"].nunique() == 1

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        result, messages = _run(analyzer.analyze_proxies)

        assert list(result) == []
        assert result.complete is False
        assert len(result.ungraded_screens) == 2
        for screen in result.ungraded_screens:
            assert math.isnan(screen.correlation)
            # A false provenance is worse than a missing one.
            assert screen.evidence["pvalue_status"] == "could_not_compute"
            assert screen.evidence["risk_level_status"] == "not_assessed"
        assert "analyze_proxies" in messages
        assert "no measurable correlation" in messages

    def test_an_unscreened_attribute_is_not_an_all_clear(self):
        df = _proxy_frame(50)
        # Sanity: the fixture really does hold the proxy the screen misses.
        assert _expected_abs_corr(df, "income", "race") > 0.95

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        result, messages = _run(analyzer.analyze_proxies)

        assert list(result) == []
        assert result.complete is False
        assert [e["protected_attribute"] for e in result.screens_not_run] == ["race"]
        assert "min_sample_size=100" in result.screens_not_run[0]["reason"]
        assert "did not run" in messages

    def test_a_zero_threshold_is_honoured_not_replaced_by_the_default(self):
        """``correlation_threshold or self.correlation_threshold`` made 0.0
        fall back to 0.3, so asking to see EVERY pair returned fewer."""
        rng = np.random.default_rng(7)
        n = 600
        race = (rng.random(n) < 0.5).astype(int)
        df = pd.DataFrame({"race": race, "weak": race * 1.0 + rng.normal(0, 4.0, n)})
        expected = _expected_abs_corr(df, "weak", "race")
        assert 0.0 < expected < 0.3

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["weak"])
        wide, _ = _run(analyzer.analyze_proxies, correlation_threshold=0.0)
        default, _ = _run(analyzer.analyze_proxies)

        assert [p.feature for p in wide] == ["weak"]
        assert wide[0].correlation == pytest.approx(expected, rel=1e-6)
        assert list(default) == []

    def test_control_healthy_data_still_finds_the_critical_proxy(self):
        """OVER-CORRECTION CONTROL."""
        df = _proxy_frame(500)
        expected = _expected_abs_corr(df, "income", "race")

        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        result, messages = _run(analyzer.analyze_proxies)

        assert [p.feature for p in result] == ["income"]
        assert result[0].risk_level.value == "critical"
        assert result[0].correlation == pytest.approx(expected, rel=1e-6)
        assert result[0].evidence["pvalue_status"] == "computed"
        assert result.complete is True
        assert result.ungraded_screens == []
        assert "no measurable correlation" not in messages
        assert "did not run" not in messages


# ---------------------------------------------------------------------------
# 3. the recommendation list a reader acts on
# ---------------------------------------------------------------------------


class TestRecommendationsDiscloseTheScreenThatDidNotRun:
    def test_an_unscreened_frame_gets_a_could_not_check_row(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(50), ["race"], feature_columns=["income", "noise"]
        )
        recommendations, messages = _run(analyzer.get_feature_recommendations)

        assert recommendations, "an empty list reads as 'nothing to do'"
        first = recommendations[0]
        assert first["priority"] == "NOT_ASSESSED"
        assert "COULD NOT CHECK" in first["action"]
        assert "min_sample_size=100" in first["rationale"]
        assert "did not run" in messages

    def test_the_could_not_check_row_reaches_the_explanation(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(50), ["race"], feature_columns=["income", "noise"]
        )
        explanation, _ = _run(analyzer.get_explanation)
        assert any("COULD NOT CHECK" in str(rec) for rec in explanation.recommendations)

    def test_control_healthy_data_recommends_the_real_finding_first(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(500), ["race"], feature_columns=["income", "noise"]
        )
        recommendations, _ = _run(analyzer.get_feature_recommendations)

        assert [r["priority"] for r in recommendations] == ["CRITICAL"]
        assert recommendations[0]["feature"] == "income"
        assert "REMOVE" in recommendations[0]["action"]
        assert not any(r["priority"] == "NOT_ASSESSED" for r in recommendations)


# ---------------------------------------------------------------------------
# 4. the surfaces that were already honest, pinned so they stay that way
# ---------------------------------------------------------------------------


class TestCorrelationSurfacesRefuseWithNaN:
    def test_get_correlation_matrix_is_nan_on_a_constant_attribute(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(500, constant_attr=True), ["race"], feature_columns=["income", "noise"]
        )
        matrix, _ = _run(analyzer.get_correlation_matrix)
        assert matrix.correlations["race"].isna().all()
        assert matrix.pvalues["race"].isna().all()

    def test_get_correlation_matrix_control(self):
        df = _proxy_frame(500)
        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        matrix, _ = _run(analyzer.get_correlation_matrix)
        assert abs(matrix.correlations.loc["income", "race"]) == pytest.approx(
            _expected_abs_corr(df, "income", "race"), rel=1e-6
        )

    def test_get_feature_correlation_matrix_is_nan_below_min_periods(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(5), ["race"], feature_columns=["income", "noise"]
        )
        matrix, _ = _run(analyzer.get_feature_correlation_matrix)
        assert matrix.isna().all().all()

    def test_get_feature_correlation_matrix_is_nan_for_a_constant_column(self):
        df = _proxy_frame(500)
        df["noise"] = 1.0
        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        matrix, _ = _run(analyzer.get_feature_correlation_matrix)
        assert math.isnan(matrix.loc["income", "noise"])
        assert matrix.loc["income", "income"] == pytest.approx(1.0)

    def test_get_feature_correlation_matrix_control(self):
        df = _proxy_frame(500)
        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        matrix, _ = _run(analyzer.get_feature_correlation_matrix)
        expected = float(np.corrcoef(df["income"], df["noise"])[0, 1])
        assert matrix.loc["income", "noise"] == pytest.approx(expected, rel=1e-9)

        with_pvalues, _ = _run(analyzer.get_feature_correlation_matrix, return_pvalues=True)
        assert isinstance(with_pvalues, tuple) and len(with_pvalues) == 2


class TestReportSerialisationCarriesTheThirdState:
    def test_to_dict_and_to_json_say_the_screen_was_incomplete(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(50), ["race"], feature_columns=["income", "noise"]
        )
        report, _ = _run(analyzer.full_analysis)
        payload = report.to_dict()
        assert payload["proxy_screen_complete"] is False
        assert len(payload["screens_not_run"]) == 1
        assert '"proxy_screen_complete": false' in report.to_json()

    def test_to_dict_control(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(500), ["race"], feature_columns=["income", "noise"]
        )
        report, _ = _run(analyzer.full_analysis)
        payload = report.to_dict()
        assert payload["proxy_screen_complete"] is True
        assert payload["screens_not_run"] == []
        assert payload["ungraded_proxy_screens"] == []
        assert [p["feature"] for p in payload["proxy_variables"]] == ["income"]
        assert '"proxy_screen_complete": true' in report.to_json()


class TestTransformIsDataNotAMeasurement:
    def test_transform_returns_feature_values_and_reduces_the_association(self):
        df = _proxy_frame(500)
        analyzer = FeatureEngineeringAnalyzer(df, ["race"], feature_columns=["income", "noise"])
        transformed, _ = _run(analyzer.transform, method="correlation_reduction")

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (500, 2)
        assert list(transformed.columns) == ["income", "noise"]
        before = _expected_abs_corr(df, "income", "race")
        after = abs(float(np.corrcoef(transformed["income"], df["race"])[0, 1]))
        assert before > 0.95 and after < 0.01

    def test_transform_refuses_an_unknown_method(self):
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(500), ["race"], feature_columns=["income", "noise"]
        )
        with pytest.raises(ValueError, match="Unknown transformation method"):
            analyzer.transform(method="no_such_method")  # type: ignore[arg-type]

    def test_get_transformation_report_is_none_before_any_fit(self):
        """Three states here are None / a result. None is not a measurement of
        'no change', and a caller can tell them apart."""
        analyzer = FeatureEngineeringAnalyzer(
            _proxy_frame(500), ["race"], feature_columns=["income", "noise"]
        )
        assert analyzer.get_transformation_report("residualization") is None

        _run(analyzer.transform, method="residualization")
        result, _ = _run(analyzer.get_transformation_report, "residualization")
        assert result is not None
        assert result.n_samples == 500
        assert result.correlation_before["race"] > result.correlation_after["race"]
