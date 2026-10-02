"""BGL3 batch preprocessing-3: does each unit refuse honestly when nothing is measurable?

Twelve units across preprocessing bias detection, feature engineering and the
canonical protected-attribute preparation, examined BY EXECUTION on inputs where
the quantity they report genuinely does not exist. Six carried the defect and are
fixed here; six already refused correctly and are pinned so the refusal cannot be
removed by a later "simplification".

Every number quoted in a docstring below was measured on 2026-09-27 by running
the real function, not read off the source.

THE SIX FINDINGS, before -> after:

1. ``FeatureCorrelationMatrix.get_high_correlations`` on a matrix whose only cell
   is NaN (5 rows, below ``MIN_SAMPLE_SIZE=10``): ``[]`` with no warning, which
   is byte-identical to a clean frame -> ``[]`` carrying
   ``pairs_not_measured == [('income', 'gender')]`` plus a warning.
2. ``compute_pearson_correlation_matrix(return_pvalues=True)`` on two columns
   sharing 5 non-null rows against ``min_periods=10``: correlations correctly
   NaN, p-values ``1.0`` in all four cells with no warning, so the documented
   ``corr[pvals < 0.05]`` usage dropped an untested pair as "not significant"
   -> NaN off the diagonal plus a warning naming the pair and its overlap.
3. ``analyze_intersectional_correlations`` on 20 rows across four
   gender x race cells (2, 5, 6, 7) at ``min_group_size=30``:
   ``within_group_correlations {}`` and ``between_group_differences {}`` with no
   warning -> ``coverage 'not_assessed'``, all four cells named in
   ``groups_not_assessed``, plus a warning.
4. the same function on 400 rows whose four group means were -99.94, -99.98,
   -1.00, -1.03: ``coefficient_of_variation 0`` beside ``max_difference 98.98``,
   so the largest gap in the frame read as zero dispersion -> 0.9798 measured,
   and NaN (never 0) when the means average to exactly 0.
5. ``check_geographic_redlining_risk`` on 600 rows over 60 ZIPs with the outcome
   column entirely NULL: ``redlining_risk_score 0.0``, ``findings []``, no
   warning; with the geographic column entirely NULL, the same 0.0; with a
   'yes'/'no' outcome column the groupby raised, the old bare ``except`` logged
   it at DEBUG and dropped ``outcome_analysis`` from the result, and the score
   was still 0.0 -> ``None`` when no component was measurable, ``'partial'``
   coverage and a named reason when only some were.
6. ``prepare_protected_attributes`` on 100 rows of ``gender=None``:
   ``usable ['gender']``, ``excluded []``, ``notes []``, because
   ``nunique(dropna=True)`` is 0 and ``0 <= 12`` -> excluded with a reason. A
   single-level column stays usable and is disclosed.
7. ``compare_to_benchmark`` on 50 rows of ``gender=None`` against a 50/50
   benchmark: ``sample_size 0`` with ``representation_ratios {'f': 0.0,
   'm': 0.0}`` and ``gaps {'f': -0.5, 'm': -0.5}``, four graded numbers over
   zero observations -> both empty, ``measurement_status 'could_not_measure'``,
   ``groups_not_measured ['f', 'm']``, plus a warning.
8. ``create_analysis_dashboard`` on those same 5 unmeasurable rows: the summary
   panel read "Total features analyzed: 1 / Protected attributes: 1 / Proxy
   Variables Found:" with nothing under the heading, while the
   ``ProxyScreenResult`` it was handed carried ``screens_not_run`` and the
   matrix was all NaN -> the panel states both gaps.

Over-correction controls sit beside every one of them: a unit that refuses
everything is as wrong as one that answers everything, and the difference is
only visible with both cases in the file.
"""

import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)

from vfairness.preprocessing.bias_detection.detector import (  # noqa: E402
    AUDIT_MODULES,
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_UNASSESSED,
    COVERAGE_UNRECORDED,
    BiasAuditReport,
    BiasDetector,
)
from vfairness.preprocessing.bias_detection.historical import (  # noqa: E402
    check_geographic_redlining_risk,
)
from vfairness.preprocessing.bias_detection.proxy import (  # noqa: E402
    compute_proxy_correlations,
)
from vfairness.preprocessing.bias_detection.representation import (  # noqa: E402
    calculate_representation_ratio,
    compare_to_benchmark,
)
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    FeatureCorrelationMatrix,
    analyze_intersectional_correlations,
    compute_feature_correlations,
    compute_pearson_correlation_matrix,
    identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.significance import (  # noqa: E402
    paired_metric_significance,
)
from vfairness.preprocessing.feature_engineering.visualization import (  # noqa: E402
    create_analysis_dashboard,
)
from vfairness.preprocessing.protected_binning import (  # noqa: E402
    prepare_protected_attributes,
)


def _caught(fn, *args, **kwargs) -> Tuple[Any, List[str]]:
    """Run ``fn`` and return ``(result, [warning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def _said(messages: List[str], needle: str) -> bool:
    return any(needle in m for m in messages)


# ── fixtures ────────────────────────────────────────────────────────────────


def _five_row_frame() -> pd.DataFrame:
    """Five rows: below MIN_SAMPLE_SIZE (10) and below the proxy screen's 100."""
    return pd.DataFrame(
        {
            "gender": ["m", "f", "m", "f", "m"],
            "income": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def _proxy_frame(n: int = 300) -> pd.DataFrame:
    """A frame where income determines gender almost perfectly."""
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "gender": np.where(g == 1, "m", "f"),
            "income": g * 50.0 + rng.normal(0, 1, n),
        }
    )


def _intersectional_frame(offset: float) -> pd.DataFrame:
    """400 rows, four gender x race cells, group means around ``offset``."""
    rng = np.random.default_rng(3)
    n = 400
    gender = rng.choice(["m", "f"], n)
    race = rng.choice(["a", "b"], n)
    base = np.where(gender == "m", offset * 100.0, offset * 1.0)
    return pd.DataFrame({"gender": gender, "race": race, "score": base + rng.normal(0, 0.5, n)})


def _zip_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    zips = rng.choice([f"{z:05d}" for z in range(10001, 10061)], n)
    inner = np.isin(zips, [f"{z:05d}" for z in range(10001, 10031)])
    approved = np.where(inner, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(int)
    return pd.DataFrame({"zip": zips, "approved": approved})


# ── 1. FeatureCorrelationMatrix.get_high_correlations ───────────────────────


class TestGetHighCorrelations:
    def test_an_unmeasured_pair_is_named_not_dropped_into_the_empty_list(self):
        """Before: `[]`, no warning, identical to a clean frame.

        The matrix for a 5-row frame holds NaN in its only cell, and
        `abs(nan) >= 0.3` is False, so the pair left no trace at all.
        """
        matrix = compute_feature_correlations(_five_row_frame(), ["gender"])
        assert not np.isfinite(matrix.correlations.loc["income", "gender"])

        result, messages = _caught(matrix.get_high_correlations, 0.3)

        assert list(result) == []
        assert result.pairs_not_measured == [("income", "gender")]
        assert result.complete is False
        assert _said(messages, "carry no measured correlation")
        assert _said(messages, "income ~ gender")

    def test_a_protected_attribute_that_is_not_a_column_is_named(self):
        """The whole column stays NaN, so nothing at all was compared."""
        matrix = compute_feature_correlations(_proxy_frame(), ["race"])
        result, messages = _caught(matrix.get_high_correlations, 0.3)

        assert list(result) == []
        assert ("income", "race") in result.pairs_not_measured
        assert _said(messages, "could-not-check")

    # ── over-correction control ──

    def test_a_measured_matrix_still_reports_its_high_correlations_silently(self):
        matrix = compute_feature_correlations(_proxy_frame(), ["gender"])
        result, messages = _caught(matrix.get_high_correlations, 0.3)

        assert [(f, a) for f, a, _v in result] == [("income", "gender")]
        assert result[0][2] > 0.9
        assert result.pairs_not_measured == []
        assert result.complete is True
        assert not _said(messages, "get_high_correlations")

    def test_the_result_is_still_a_plain_list_to_every_existing_consumer(self):
        """Consumers unpack 3-tuples, call len() and isinstance(x, list) on this."""
        matrix = compute_feature_correlations(_proxy_frame(), ["gender"])
        result = matrix.get_high_correlations(0.0)

        assert isinstance(result, list)
        assert len(result) == 1
        for feature, attr, value in result:
            assert isinstance(feature, str) and isinstance(attr, str)
            assert isinstance(float(value), float)


# ── 2. compute_pearson_correlation_matrix ───────────────────────────────────


def _sparse_overlap_frame() -> pd.DataFrame:
    """Two numeric columns sharing exactly 5 non-null rows."""
    return pd.DataFrame(
        {
            "a": [1.0, 2, 3, 4, 5] + [np.nan] * 7,
            "b": [2.0, 1, 4, 3, 6] + [np.nan] * 7,
        }
    )


class TestPearsonMatrixPvalues:
    def test_a_pair_below_min_periods_gets_no_pvalue_rather_than_one(self):
        """Before: pvalue matrix was all 1.0, including the untested a ~ b pair.

        A p-value of 1.0 is the strongest "no evidence of association" the test
        can state, and it was the matrix's initial value, so every skipped pair
        kept it. The correlation matrix beside it correctly said NaN.
        """
        (corr, pvals), messages = _caught(
            compute_pearson_correlation_matrix, _sparse_overlap_frame(), return_pvalues=True
        )

        assert not np.isfinite(corr.loc["a", "b"])
        assert not np.isfinite(pvals.loc["a", "b"]), "an untested pair must not carry a p-value"
        assert not np.isfinite(pvals.loc["b", "a"])
        assert _said(messages, "min_periods=10")
        assert _said(messages, "a ~ b (n=5)")

    def test_the_documented_significance_filter_no_longer_reads_untested_as_clean(self):
        """`corr[pvals < 0.05]` is the docstring's own usage.

        With 1.0 in the cell the untested pair was filtered out silently, which
        is the same output as a pair tested and found insignificant. With NaN it
        is still filtered out, but `pvals.isna()` now names it.
        """
        (_corr, pvals), _messages = _caught(
            compute_pearson_correlation_matrix, _sparse_overlap_frame(), return_pvalues=True
        )
        untested = [
            (i, j)
            for i in pvals.index
            for j in pvals.columns
            if i != j and not np.isfinite(pvals.loc[i, j])
        ]
        assert untested == [("a", "b"), ("b", "a")]

    # ── over-correction control ──

    def test_a_measured_pair_still_gets_a_real_pvalue(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.normal(size=100)})
        df["b"] = df["a"] * 2 + rng.normal(0, 0.1, 100)

        (corr, pvals), messages = _caught(
            compute_pearson_correlation_matrix, df, return_pvalues=True
        )

        assert corr.loc["a", "b"] > 0.9
        assert np.isfinite(pvals.loc["a", "b"])
        assert pvals.loc["a", "b"] < 0.05
        assert not _said(messages, "min_periods")

    def test_the_diagonal_of_a_populated_column_is_a_measurement_not_a_refusal(self):
        """A column against itself needs no test: r is 1 and p is 0 by construction.

        Withholding here would be the over-correction, so it is pinned.
        """
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
        _corr, pvals = compute_pearson_correlation_matrix(df, return_pvalues=True)

        assert pvals.loc["a", "a"] == 0.0
        assert pvals.loc["b", "b"] == 0.0


# ── 3 and 4. analyze_intersectional_correlations ────────────────────────────


class TestIntersectionalCorrelations:
    def test_no_group_reaching_the_minimum_is_disclosed_not_an_empty_dict(self):
        """Before: both result dicts `{}` with no warning, on 20 rows.

        Measured group sizes were 2, 5, 6 and 7 against min_group_size=30, so
        nothing was examined, and `{}` was the same output a frame whose groups
        were all examined and all agreed would produce.
        """
        rng = np.random.default_rng(3)
        n = 20
        df = pd.DataFrame(
            {
                "gender": rng.choice(["m", "f"], n),
                "race": rng.choice(["a", "b"], n),
                "income": rng.normal(50000, 1000, n),
            }
        )

        result, messages = _caught(
            analyze_intersectional_correlations, df, ["gender", "race"], min_group_size=30
        )

        assert result["intersectional_groups"] == []
        assert result["within_group_correlations"] == {}
        assert result["between_group_differences"] == {}
        assert result["coverage"] == "not_assessed"
        assert len(result["groups_not_assessed"]) == 4
        assert all("min_group_size=30" in e["reason"] for e in result["groups_not_assessed"])
        assert _said(messages, "were NOT examined")
        assert _said(messages, "not a finding that")

    def test_a_negative_mean_no_longer_makes_the_widest_gap_read_as_zero_spread(self):
        """Before: coefficient_of_variation 0 beside max_difference 98.98.

        Group means were -99.94, -99.98, -1.00, -1.03. `np.mean(means) > 0` is
        False for them, so the literal-0 fallback fired, and 0 dispersion is
        perfect agreement between the groups: the single largest between-group
        gap in the frame read as no variation at all.
        """
        result, messages = _caught(
            analyze_intersectional_correlations,
            _intersectional_frame(-1.0),
            ["gender", "race"],
            min_group_size=30,
        )
        diff = result["between_group_differences"]["score"]

        assert diff["max_difference"] > 90.0
        assert np.isfinite(diff["coefficient_of_variation"])
        assert diff["coefficient_of_variation"] > 0.5, (
            "a 99-unit gap between group means cannot have near-zero dispersion"
        )
        assert result["coverage"] == "complete"
        assert not _said(messages, "coefficient of variation")

    def test_means_averaging_to_zero_report_no_coefficient_of_variation(self):
        """The one case where a CV genuinely does not exist: nan, never 0."""
        rng = np.random.default_rng(9)
        n = 400
        gender = rng.choice(["m", "f"], n)
        race = rng.choice(["a", "b"], n)
        # Group means of exactly +50 and -50 average to 0, so std/mean is 0/0.
        df = pd.DataFrame(
            {"gender": gender, "race": race, "score": np.where(gender == "m", 50.0, -50.0)}
        )

        result, messages = _caught(
            analyze_intersectional_correlations, df, ["gender", "race"], min_group_size=30
        )
        diff = result["between_group_differences"]["score"]

        assert diff["max_difference"] == pytest.approx(100.0)
        assert np.isnan(diff["coefficient_of_variation"])
        assert _said(messages, "no coefficient of variation exists")

    # ── over-correction control ──

    def test_a_healthy_frame_still_reports_a_measured_coefficient_silently(self):
        """The positive-mean twin of the finding above.

        The two frames are mirror images (means +100/+1 against -100/-1), so the
        same spread must read the same whichever side of zero it sits on. Before
        the fix the positive one measured 0.9798 and the negative one 0, which is
        how a sign test hid the largest gap in the frame.
        """
        result, messages = _caught(
            analyze_intersectional_correlations,
            _intersectional_frame(1.0),
            ["gender", "race"],
            min_group_size=30,
        )
        mirrored, _m = _caught(
            analyze_intersectional_correlations,
            _intersectional_frame(-1.0),
            ["gender", "race"],
            min_group_size=30,
        )
        diff = result["between_group_differences"]["score"]

        assert result["coverage"] == "complete"
        assert result["groups_not_assessed"] == []
        assert len(result["intersectional_groups"]) == 4
        assert diff["coefficient_of_variation"] == pytest.approx(0.9798, abs=0.01)
        assert diff["coefficient_of_variation"] == pytest.approx(
            mirrored["between_group_differences"]["score"]["coefficient_of_variation"], abs=0.01
        )
        assert not _said(messages, "analyze_intersectional_correlations")

    def test_partial_coverage_names_only_the_groups_it_could_not_examine(self):
        """A rare cell beside three large ones is 'partial', not 'not_assessed'."""
        rng = np.random.default_rng(11)
        n = 400
        gender = np.where(np.arange(n) < 3, "x", rng.choice(["m", "f"], n))
        race = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"gender": gender, "race": race, "score": rng.normal(100, 5, n)})

        result, messages = _caught(
            analyze_intersectional_correlations, df, ["gender", "race"], min_group_size=30
        )

        assert result["coverage"] == "partial"
        assert result["intersectional_groups"], "the large cells must still be examined"
        assert [e["group"] for e in result["groups_not_assessed"]]
        assert all(g.startswith("x_") for g in [e["group"] for e in result["groups_not_assessed"]])
        assert _said(messages, "were NOT examined")


# ── 5. check_geographic_redlining_risk ──────────────────────────────────────


class TestRedliningRisk:
    def test_an_all_null_outcome_column_is_named_not_scored_as_no_disparity(self):
        """Before: redlining_risk_score 0.0, findings [], no warning at all.

        `outcome_range > 0.3` is False for a NaN, so an outcome column with not
        one observation took the same branch as a measured range of 0.05. The
        score is a SUM, so the missing 0.4 left 0.0 behind, which reads as no
        redlining risk.
        """
        df = _zip_frame()
        df["approved"] = np.nan

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == "partial"
        reasons = [e["component"] for e in result["components_not_assessed"]]
        assert reasons == ["outcome_disparity"]
        assert _said(messages, "LOWER BOUND")
        assert _said(messages, "outcome_disparity")

    def test_an_all_null_geographic_column_reports_no_score_at_all(self):
        """Before: 0.0. Nothing about the geography could be read, so there is no score."""
        df = _zip_frame()
        df["zip"] = np.nan

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["redlining_risk_score"] is None
        assert result["assessment_coverage"] == "not_assessed"
        assert _said(messages, "not one component")
        assert _said(messages, "NOT 0.0")

    def test_a_non_numeric_outcome_column_names_the_refusal_instead_of_a_debug_log(self):
        """Before: the groupby raised, the bare `except` logged at DEBUG, and
        `outcome_analysis` was absent from the result with the score still 0.0.
        """
        df = _zip_frame()
        df["approved"] = np.where(df["approved"] == 1, "yes", "no")

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == "partial"
        assert [e["component"] for e in result["components_not_assessed"]] == ["outcome_disparity"]
        assert "could not be computed" in result["components_not_assessed"][0]["reason"]
        assert _said(messages, "LOWER BOUND")

    def test_a_missing_geographic_column_reports_no_score(self):
        result, _messages = _caught(check_geographic_redlining_risk, _zip_frame(), "not_a_column")

        assert result["redlining_risk_score"] is None
        assert result["assessment_coverage"] == "not_assessed"

    # ── over-correction control ──

    def test_a_real_geographic_disparity_is_still_scored_silently(self):
        result, messages = _caught(
            check_geographic_redlining_risk, _zip_frame(), "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == COVERAGE_COMPLETE
        assert result["components_not_assessed"] == []
        assert result["redlining_risk_score"] == pytest.approx(0.4)
        assert any("outcome disparity" in f for f in result["findings"])
        assert not _said(messages, "check_geographic_redlining_risk")


# ── 6. prepare_protected_attributes ─────────────────────────────────────────


class TestPrepareProtectedAttributes:
    def test_an_all_null_protected_column_is_excluded_with_a_reason(self):
        """Before: usable ['gender'], excluded [], notes [], on 100 empty rows.

        `nunique(dropna=True)` is 0 and `0 <= _MAX_GROUP_CARDINALITY` is true,
        so the low-cardinality branch claimed a column that yields no group.
        """
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"gender": [None] * 100, "race": rng.choice(["a", "b"], 100)})

        prep = prepare_protected_attributes(df, ["gender", "race"])

        assert prep.usable == ["race"]
        assert [col for col, _reason in prep.excluded] == ["gender"]
        assert "no values at all" in dict(prep.excluded)["gender"]

    def test_a_single_level_column_stays_usable_and_says_it_carries_no_comparison(self):
        """One level is one group, and a fairness comparison needs two.

        Before: usable ['gender'] with excluded [] and notes [], so nothing
        anywhere stated that no between-group comparison exists for it.
        """
        prep = prepare_protected_attributes(pd.DataFrame({"gender": ["f"] * 100}), ["gender"])

        assert prep.usable == ["gender"], "excluding it would be the over-correction"
        assert prep.excluded == []
        labels = [label for label, _detail in prep.notes]
        assert "Coverage" in labels
        assert any("single group" in detail for _label, detail in prep.notes)

    # ── over-correction control ──

    def test_a_real_two_level_attribute_is_usable_with_nothing_disclosed(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"gender": rng.choice(["m", "f"], 100)})

        prep = prepare_protected_attributes(df, ["gender"])

        assert prep.usable == ["gender"]
        assert prep.excluded == []
        assert prep.notes == []


# ── 7. compare_to_benchmark ─────────────────────────────────────────────────


class TestCompareToBenchmark:
    def test_zero_observed_rows_report_no_ratio_rather_than_a_ratio_of_zero(self):
        """Before: sample_size 0 with representation_ratios {'f': 0.0, 'm': 0.0}
        and gaps {'f': -0.5, 'm': -0.5}: four graded numbers over no observations.

        A ratio of 0 is the severest finding this scale carries, not a small one,
        and `calculate_representation_ratio` one function below already refuses
        the identical case with None and a warning.
        """
        df = pd.DataFrame({"gender": [None] * 50})

        result, messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["sample_size"] == 0
        assert dict(result["representation_ratios"]) == {}
        assert dict(result["gaps"]) == {}
        assert result["measurement_status"] == "could_not_measure"
        assert result["groups_not_measured"] == ["f", "m"]
        assert _said(messages, "no group share exists")
        assert _said(messages, "NOT a ratio of 0")

    # ── over-correction controls ──

    def test_a_measured_benchmark_comparison_is_unchanged(self):
        result, messages = _caught(
            compare_to_benchmark, _proxy_frame(), "gender", {"m": 0.5, "f": 0.5}
        )

        assert result["measurement_status"] == "measured"
        assert result["groups_not_measured"] == []
        assert result["representation_ratios"]["f"] == pytest.approx(
            result["actual_distribution"]["f"] / 0.5
        )
        assert np.isfinite(result["pvalue"])
        assert not _said(messages, "no group share exists")

    def test_a_group_absent_from_observed_rows_still_gets_a_measured_zero(self):
        """The distinction that matters: 0% of 300 observed rows IS a finding."""
        df = pd.DataFrame({"gender": ["m"] * 300})

        result, _messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["sample_size"] == 300
        assert result["measurement_status"] == "measured"
        assert result["representation_ratios"]["f"] == 0.0
        assert result["gaps"]["f"] == pytest.approx(-0.5)


# ── 8. create_analysis_dashboard ────────────────────────────────────────────


def _panel_text(fig) -> str:
    return "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)


class TestAnalysisDashboard:
    @needs_matplotlib
    def test_the_summary_panel_states_what_was_never_measured(self):
        """Before: "Total features analyzed: 1 / Protected attributes: 1 / Proxy
        Variables Found:" with nothing under the heading, on 5 unmeasurable rows.

        Both inputs carried the coverage record. `correlations` was NaN in its
        only cell and `identify_proxy_variables` returned a ProxyScreenResult
        whose `screens_not_run` named the pair it never looked at. The panel read
        neither, so it was the surface where the disclosure died.
        """
        df = _five_row_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            proxies = identify_proxy_variables(df, ["gender"])
            fig = create_analysis_dashboard(matrix, proxies)

        text = _panel_text(fig)
        assert "NOT MEASURED: 1 of 1" in text
        assert "NOT SCREENED: 1 pair(s)" in text
        assert "gender ~ (whole attribute)" in text
        assert "none found, on incomplete coverage" in text
        assert "Features analyzed: 0 of 1" in text

    @needs_matplotlib
    def test_a_matrix_whose_coverage_cannot_be_read_says_so_rather_than_not_measured(self):
        """The third state. A hand-built matrix can declare a protected attribute
        that is not a column of its own `correlations` frame, and then how much
        was measured is UNKNOWN, which is neither measured nor unmeasured.
        Claiming "NOT MEASURED" about it would be the same fabrication in the
        other direction.
        """
        features = ["zipcode", "income"]
        matrix = FeatureCorrelationMatrix(
            correlations=pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=features, columns=features),
            pvalues=pd.DataFrame(np.full((2, 2), 0.01), index=features, columns=features),
            feature_names=features,
            protected_attributes=["race"],
            method="pearson",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = create_analysis_dashboard(matrix, [])

        text = _panel_text(fig)
        assert "COVERAGE UNKNOWN" in text
        assert "NOT MEASURED" not in text
        assert "none found, coverage unknown" in text

    # ── over-correction control ──

    @needs_matplotlib
    def test_a_measured_dashboard_carries_the_finding_and_no_coverage_warning(self):
        df = _proxy_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            proxies = identify_proxy_variables(df, ["gender"])
            fig = create_analysis_dashboard(matrix, proxies)

        text = _panel_text(fig)
        assert "Features analyzed: 1 of 1" in text
        assert "NOT MEASURED" not in text
        assert "NOT SCREENED" not in text
        assert "CRITICAL: 1" in text


# ── 9 to 12. the units that already refused correctly ───────────────────────


class TestUnitsAlreadyCorrect:
    """Pins, not fixes. Each of these was executed on an unmeasurable input and
    already returned a could-not-check state with a warning naming the gap. They
    are pinned because the four defects above show how easily such a state is
    flattened back into a clean-looking value by a later edit."""

    def test_compute_proxy_correlations_refuses_a_single_level_attribute(self):
        """Measured: primary_correlation nan, risk_level None,
        measurement_status 'could_not_measure', plus a NOT SCREENED warning."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"gender": ["m"] * 60, "income": rng.normal(0, 1, 60)})

        result, messages = _caught(compute_proxy_correlations, df, "income", "gender")

        assert np.isnan(result["primary_correlation"])
        assert result["risk_level"] is None
        assert result["measurement_status"] == "could_not_measure"
        assert _said(messages, "NOT SCREENED")

    def test_compute_proxy_correlations_still_finds_a_real_proxy(self):
        result, _messages = _caught(compute_proxy_correlations, _proxy_frame(), "income", "gender")

        assert result["measurement_status"] == "measured"
        assert result["primary_correlation"] > 0.9
        assert result["risk_level"] == "critical"

    def test_calculate_representation_ratio_refuses_an_all_null_column(self):
        """Measured: every graded field None, plus an UNMEASURED warning."""
        df = pd.DataFrame({"gender": [None] * 50})

        result, messages = _caught(calculate_representation_ratio, df, "gender", "f", 0.5)

        assert result["actual_proportion"] is None
        assert result["representation_ratio"] is None
        assert result["is_underrepresented"] is None
        assert result["is_overrepresented"] is None
        assert _said(messages, "UNMEASURED (None), not zero")

    def test_calculate_representation_ratio_still_measures_a_real_column(self):
        result, messages = _caught(
            calculate_representation_ratio, _proxy_frame(), "gender", "f", 0.5
        )

        assert result["representation_ratio"] == pytest.approx(result["actual_proportion"] / 0.5)
        assert result["is_underrepresented"] in (True, False)
        assert not _said(messages, "UNMEASURED")

    def test_identify_proxy_variables_records_the_pairs_it_never_screened(self):
        """Measured: [] with screens_not_run naming the attribute and a warning."""
        result, messages = _caught(identify_proxy_variables, _five_row_frame(), ["gender"])

        assert list(result) == []
        assert result.complete is False
        assert result.screens_not_run[0]["protected_attribute"] == "gender"
        assert "min_sample_size=100" in result.screens_not_run[0]["reason"]
        assert _said(messages, "NOT SCREENED")

    def test_identify_proxy_variables_still_flags_a_real_critical_proxy(self):
        result, _messages = _caught(identify_proxy_variables, _proxy_frame(), ["gender"])

        assert result.complete is True
        assert [p.risk_level.value for p in result] == ["critical"]

    def test_paired_metric_significance_refuses_one_protected_group(self):
        """Measured: p_value nan, significant None, interpretation NOT ASSESSED."""
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 60)
        p = rng.integers(0, 2, 60)
        s = np.array(["A"] * 60)

        result, messages = _caught(paired_metric_significance, y, p, s, y, p, s, n_bootstrap=50)

        assert np.isnan(result["p_value"])
        assert result["significant"] is None
        assert result["n_bootstrap_measured"] == 0
        assert "NOT ASSESSED" in result["interpretation"]
        assert _said(messages, "could not check")

    def test_paired_metric_significance_still_measures_a_real_improvement(self):
        rng = np.random.default_rng(7)
        n = 300
        s = rng.choice(["A", "B"], n)
        y = rng.integers(0, 2, n)
        before = np.where(s == "A", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
        after = np.where(s == "A", rng.random(n) < 0.55, rng.random(n) < 0.5).astype(int)

        result, _messages = _caught(
            paired_metric_significance, y, before, s, y, after, s, n_bootstrap=200
        )

        assert np.isfinite(result["p_value"])
        assert result["significant"] is True
        assert result["delta"] < 0
        assert "improved (fairer)" in result["interpretation"]


def _hand_built_report(**over) -> BiasAuditReport:
    base = dict(
        timestamp="2026-09-27",
        dataset_info={},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return BiasAuditReport(**base)


class TestEmptyIsNotAMeasurement:
    """``BiasAuditReport.empty_is_not_a_measurement`` already answers every state
    the library's own producer can reach. ``full_audit`` always records both
    ``modules_run`` and ``attribute_assessed``, so the four cases below are the
    ones a reader meets.

    RESIDUAL, recorded rather than changed: a HAND-BUILT report that records
    ``modules_run`` and neither attribute field answers ``None`` here, which
    means "the zero is a measurement", although whether anything was assessed is
    unknown. ``assessment_coverage()`` does return ``"unrecorded"`` for it, so
    the state is reachable, and ``execution_coverage``'s own comment names this
    as a deliberate backward-compatibility choice for reports predating the
    record. Pinned below as the state it is, not as the state it should be.
    """

    def test_a_two_row_frame_withholds_the_count(self):
        """Measured on the real detector: 'ran_but_assessed_nothing', count None."""
        rng = np.random.default_rng(6)
        n = 500
        g = rng.choice(["m", "f"], n)
        df = pd.DataFrame(
            {
                "gender": g,
                "zipcode": rng.choice([f"{z:05d}" for z in range(10001, 10011)], n),
                "approved": np.where(g == "m", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(
                    int
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = BiasDetector(
                df.head(2), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit()

        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert "none of them assessed" in report.empty_is_not_a_measurement()
        count, messages = _caught(report.get_critical_count)
        assert count is None
        assert _said(messages, "would not be a measurement")

    def test_no_module_run_is_a_reason_not_a_measurement(self):
        report = _hand_built_report(modules_run=[])

        assert report.execution_coverage() == COVERAGE_NONE
        assert report.empty_is_not_a_measurement() == "no audit module executed"

    def test_an_unrecorded_run_is_a_reason_not_a_measurement(self):
        report = _hand_built_report()

        assert report.execution_coverage() == COVERAGE_UNRECORDED
        assert "does not record" in report.empty_is_not_a_measurement()

    def test_all_modules_run_with_nothing_assessed_is_a_reason(self):
        report = _hand_built_report(
            modules_run=list(AUDIT_MODULES),
            attribute_observations={"gender": 2},
            attribute_assessed={"gender": False},
        )

        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert "gender" in report.empty_is_not_a_measurement()

    def test_the_residual_unrecorded_assessment_state_is_now_refused_too(self):
        """THE RESIDUAL IS CLOSED. This test recorded a state that was not refused.

        It was named ..._is_reachable_by_a_caller and asserted
        ``empty_is_not_a_measurement() is None``, documenting that a report
        recording every module and no assessment half at all slipped through the
        refusal, with assessment_coverage() as "the caller's only way to tell this
        apart from a real measurement". That was an honest record of an open gap
        rather than a claim it was fine.

        Closed 2026-09-27. The method now returns a reason for that state, so the
        caller does not have to cross-reference a second method to notice. The
        subject is kept and the assertion is inverted; the name changed because it
        had become false.
        """
        report = _hand_built_report(modules_run=list(AUDIT_MODULES))

        reason = report.empty_is_not_a_measurement()
        assert reason is not None, (
            "the unrecorded assessment half is reachable again and no longer "
            "refused, so an empty finding list from it reads as a clean bill"
        )
        assert "does not record" in reason, reason
        assert report.assessment_coverage() == COVERAGE_UNRECORDED, (
            "and the second channel still says which silence this is"
        )

    # ── over-correction control ──

    def test_a_complete_audit_of_a_real_frame_measures_its_findings(self):
        rng = np.random.default_rng(6)
        n = 500
        g = rng.choice(["m", "f"], n)
        df = pd.DataFrame(
            {
                "gender": g,
                "zipcode": rng.choice([f"{z:05d}" for z in range(10001, 10011)], n),
                "approved": np.where(g == "m", rng.random(n) < 0.8, rng.random(n) < 0.3).astype(
                    int
                ),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = BiasDetector(
                df, protected_attributes=["gender"], outcome_column="approved"
            ).full_audit()

        assert report.execution_coverage() == COVERAGE_COMPLETE
        assert report.empty_is_not_a_measurement() is None
        count, messages = _caught(report.get_critical_count)
        assert count == len(report.critical_issues)
        assert not _said(messages, "get_critical_count")
