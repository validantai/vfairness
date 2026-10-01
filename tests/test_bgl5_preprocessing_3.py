"""BGL5 batch A-preprocessing-3: the ten overturned grades, closed and pinned.

An independent audit overturned ten grades in this batch. Every one had the same
shape: a quantity the code could not measure reached a reader as a number, a band
or a bar, and the refusal that existed reached only the one degenerate input the
previous pin had tried. This file pins the corrected behaviour BY EXECUTION, and
every number in it was measured on 2026-09-27 by running the real function.

Each class pins one unit and carries its own OVER-CORRECTION CONTROL, asserted
with the actual measured value, because a unit that refuses everything passes
every refusal test and destroys the library.

The two figure-shaped fixes are checked by reading the ARTISTS (``ax.patches``
and their widths, ``ax.texts``), never by reading tick values: an empty axes still
carries a "0.0" tick, so a tick-based check reads false in both directions. The
tick LABELS that are asserted are ones the fix sets explicitly.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from vfairness.preprocessing.bias_detection.historical import (  # noqa: E402
    attribute_historical_pattern,
    check_geographic_redlining_risk,
    domain_historical_context,
)
from vfairness.preprocessing.bias_detection.representation import (  # noqa: E402
    calculate_representation_ratio,
    compare_to_benchmark,
)
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    analyze_intersectional_correlations,
    compute_feature_correlations,
    identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.significance import (  # noqa: E402
    paired_metric_significance,
)
from vfairness.preprocessing.feature_engineering.visualization import (  # noqa: E402
    create_analysis_dashboard,
    plot_correlation_bars,
    plot_intersectional_heatmap,
)
from vfairness.preprocessing.protected_binning import (  # noqa: E402
    prepare_protected_attributes,
)


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def _said(messages, fragment: str) -> bool:
    return any(fragment in m for m in messages)


def _zip_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    zips = rng.choice([f"{z:05d}" for z in range(10001, 10061)], n)
    inner = np.isin(zips, [f"{z:05d}" for z in range(10001, 10031)])
    approved = np.where(inner, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(int)
    return pd.DataFrame({"zip": zips, "approved": approved})


def _proxy_frame(n: int = 300) -> pd.DataFrame:
    """'income' determines 'gender'; 'age' is constant, so its eta is NaN."""
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "gender": np.where(g == 1, "m", "f"),
            "income": g * 50.0 + rng.normal(0, 1, n),
            "age": np.full(n, 40.0),
        }
    )


def _intersectional_frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "g": rng.choice(["m", "f"], n),
            "r": rng.choice(["a", "b"], n),
            "score": rng.normal(50, 5, n),
        }
    )


# ── 1. check_geographic_redlining_risk ──────────────────────────────────────


class TestRedliningNeedsTwoObservedGeographicUnits:
    """A RANGE IS A COMPARISON, SO IT NEEDS TWO UNITS.

    ``Series.max()`` and ``Series.min()`` skip NaN, so the previous finiteness
    guard was satisfied by a single observed geographic unit and the refusal
    reached only a column that was null in every unit.
    """

    def test_one_observed_zip_of_sixty_is_not_a_measured_range(self):
        """Measured: 13 of 600 rows observed, in 1 of 60 ZIPs.

        Before: assessment_coverage 'complete', components_not_assessed [],
        redlining_risk_score 0.0, outcome_range 0.0 and no warning.
        """
        df = _zip_frame()
        keep = df["zip"] == df["zip"].iloc[0]
        df.loc[~keep, "approved"] = np.nan
        assert int(df["approved"].notna().sum()) == 13
        assert df.loc[df["approved"].notna(), "zip"].nunique() == 1

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == "partial"
        assert [e["component"] for e in result["components_not_assessed"]] == ["outcome_disparity"]
        assert "finite over 1 of 60" in result["components_not_assessed"][0]["reason"]
        # The fabricated number is gone from the surface that held it.
        assert result["outcome_analysis"]["outcome_range"] is None
        assert result["outcome_analysis"]["n_geographic_units"] == 60
        assert result["outcome_analysis"]["n_geographic_units_with_a_finite_mean"] == 1
        assert _said(messages, "LOWER BOUND")

    def test_two_observed_units_of_two_hundred_is_a_range_but_partial_coverage(self):
        """Measured on the same 600 rows with 30 of 60 ZIPs observed: the range
        IS a comparison, and it is taken over half the geography, so the score
        stays 0.4 and the coverage says partial rather than complete.
        """
        df = _zip_frame()
        first_half = sorted(df["zip"].unique())[:30]
        df.loc[~df["zip"].isin(first_half), "approved"] = np.nan

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == "partial"
        assert result["components_not_assessed"] == []
        assert [e["component"] for e in result["components_partially_assessed"]] == [
            "outcome_disparity"
        ]
        assert result["redlining_risk_score"] == pytest.approx(0.4)
        assert result["outcome_analysis"]["n_geographic_units_with_a_finite_mean"] == 30
        assert _said(messages, "measured over PART of the geography")

    def test_a_single_geographic_unit_reports_no_range_at_all(self):
        """One unit cannot differ from itself. Before: outcome_range 0.0,
        coverage 'complete' and a 'Very few geographic units (1)' finding beside
        a score of 0.0."""
        df = pd.DataFrame({"zip": ["10001"] * 200, "approved": [1] * 100 + [0] * 100})

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        assert result["outcome_analysis"]["outcome_range"] is None
        assert result["assessment_coverage"] == "partial"
        assert _said(messages, "needs at least 2")

    # ── over-correction control ──

    def test_control_a_fully_observed_geography_still_scores_its_disparity(self):
        result, messages = _caught(
            check_geographic_redlining_risk, _zip_frame(), "zip", outcome_column="approved"
        )

        assert result["assessment_coverage"] == "complete"
        assert result["components_not_assessed"] == []
        assert result["components_partially_assessed"] == []
        assert result["redlining_risk_score"] == pytest.approx(0.4)
        assert result["outcome_analysis"]["outcome_range"] == pytest.approx(1.0)
        assert result["outcome_analysis"]["n_geographic_units_with_a_finite_mean"] == 60
        assert any("outcome disparity" in f for f in result["findings"])
        assert not _said(messages, "check_geographic_redlining_risk")


# ── 2 and 3. attribute_historical_pattern / domain_historical_context ────────


class TestHistoricalPrecedentIsMatchedWholeToken:
    """A LEGAL PRECEDENT MAY NOT BE ATTACHED BY SUBSTRING.

    Both lookups resolved by ``alias in key`` over a raw lowercased string, and
    what they return on a match carries REAL citations, which is the form a
    reader cannot audit. ``protected_binning``'s module docstring records this
    exact bug class ('"age" matching "primary_language"') as a past incident.
    """

    @pytest.mark.parametrize(
        "column",
        [
            "primary_language",
            "average_salary",
            "message_count",
            "percentage_complete",
            "usage_tier",
            "package_weight",
            "storage_class",
            "triage_score",
        ],
    )
    def test_a_name_merely_containing_the_letters_age_gets_no_precedent(self, column):
        """Before: every one of these came back pattern_id 'hiring_age',
        attribute_class 'age', citing the ADEA."""
        assert attribute_historical_pattern(column, "hiring") is None

    def test_a_contract_column_gets_no_redlining_precedent(self):
        """'contract_type' contains 'tract'. Before: pattern_id
        'lending_geographic_redlining' citing Rothstein's 'The Color of Law'."""
        assert attribute_historical_pattern("contract_type", "lending") is None

    @pytest.mark.parametrize(
        "domain",
        [
            "chronic disease management",
            "threat detection",
            "anthropology research",
            "shrinkage analytics",
            "xhrx",
        ],
    )
    def test_a_domain_merely_containing_hr_gets_no_hiring_precedent(self, domain):
        """Before: all five resolved to domain 'hiring' and came back carrying
        the Amazon recruiting-tool citation, because 'hr' is a substring of
        'chronic', 'threat', 'anthropology', 'shrinkage' and 'xhrx'."""
        assert domain_historical_context(domain) is None

    def test_a_bailout_domain_gets_no_recidivism_precedent(self):
        """Before: domain 'justice', carrying the COMPAS finding, because 'bail'
        is a substring of 'bailout'."""
        assert domain_historical_context("bailout underwriting") is None

    def test_a_mortgage_phrase_resolves_to_lending_not_to_justice(self):
        """'mortgage bail-in' does hold 'bail' as a whole token, so whole-token
        matching alone is not enough: the longest match wins, and 'mortgage' is
        the lending alias this table was missing. Before: 'justice'."""
        out = domain_historical_context("mortgage bail-in")

        assert out is not None
        assert out["domain"] == "lending"

    # ── over-correction controls ──

    @pytest.mark.parametrize("column", ["age", "age_group", "customerAge", "AGE"])
    def test_control_a_real_age_column_still_gets_the_adea_precedent(self, column):
        out = attribute_historical_pattern(column, "hiring")

        assert out is not None
        assert out["pattern_id"] == "hiring_age"
        assert out["attribute_class"] == "age"
        assert "US Age Discrimination in Employment Act (ADEA)" in out["citations"]

    def test_control_real_geographic_and_race_columns_still_resolve(self):
        assert (
            attribute_historical_pattern("zip_code", "lending")["pattern_id"]
            == "lending_geographic_redlining"
        )
        assert (
            attribute_historical_pattern("census_tract_id", "lending")["attribute_class"]
            == "geographic"
        )
        assert (
            attribute_historical_pattern("race_ethnicity", "lending")["pattern_id"]
            == "lending_race_redlining"
        )

    @pytest.mark.parametrize(
        "domain,expected",
        [
            ("hiring", "hiring"),
            ("hr", "hiring"),
            ("hr analytics", "hiring"),
            ("online hiring platform", "hiring"),
            ("employment screening", "hiring"),
            ("credit scoring", "lending"),
            ("loan origination", "lending"),
            ("clinical decision support", "healthcare"),
            ("bail", "justice"),
            ("recidivism prediction", "justice"),
            ("university admissions", "education"),
            ("health insurance", "insurance"),
        ],
    )
    def test_control_a_real_domain_still_resolves(self, domain, expected):
        """'online hiring platform' is the example the old code's own comment
        claimed and never delivered: measured before the fix, it returned None,
        because the alias table holds no entry for the canonical key 'hiring'."""
        out = domain_historical_context(domain)

        assert out is not None, domain
        assert out["domain"] == expected


# ── 4. compare_to_benchmark ─────────────────────────────────────────────────


class TestBenchmarkNamesEveryGroupItReportedNoRatioFor:
    """``groups_not_measured`` was ``[] if measured else groups_seen``, and
    ``measured`` was ``n_total > 0``, so the field reported "nothing unmeasured"
    for every input with at least one observed row."""

    @pytest.mark.parametrize("expected", [0.0, -0.1, float("nan")])
    def test_a_group_with_no_divisible_expected_share_is_named(self, expected):
        """Measured on 300 rows against {'m': 0.5, 'f': 0.5, 'x': <expected>}.

        Before: representation_ratios {'f': 1.0, 'm': 1.0}, gaps {'f': 0.0,
        'm': 0.0}, measurement_status 'measured', groups_not_measured [] and no
        warning, with 'x' dropped by ``expected_prop > 0``.
        """
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        result, messages = _caught(
            compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5, "x": expected}
        )

        assert result["groups_not_measured"] == ["x"]
        assert result["measurement_status"] == "partial"
        assert "x" not in dict(result["representation_ratios"])
        assert _said(messages, "NOT a ratio of 0")
        # The groups that WERE divisible keep their real numbers.
        assert dict(result["representation_ratios"]) == {"f": 1.0, "m": 1.0}

    def test_an_observed_group_the_benchmark_never_mentions_is_named_too(self):
        """20 of 300 rows are 'nb'. Before: absent from both dicts with
        groups_not_measured [] and measurement_status 'measured', while the
        chi-squared half DID disclose it."""
        df = pd.DataFrame({"gender": ["m"] * 140 + ["f"] * 140 + ["nb"] * 20})

        result, messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["groups_not_measured"] == ["nb"]
        assert result["measurement_status"] == "partial"
        assert result["pvalue"] is None
        assert _said(messages, "the benchmark never")

    def test_no_observed_row_still_reports_could_not_measure(self):
        df = pd.DataFrame({"gender": [None] * 50})

        result, messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["measurement_status"] == "could_not_measure"
        assert result["groups_not_measured"] == ["f", "m"]
        assert dict(result["representation_ratios"]) == {}
        assert _said(messages, "no group share exists")

    # ── over-correction controls ──

    def test_control_a_fully_divisible_benchmark_reports_measured_and_nothing_missing(self):
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        result, messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["measurement_status"] == "measured"
        assert result["groups_not_measured"] == []
        assert dict(result["representation_ratios"]) == {"f": 1.0, "m": 1.0}
        assert result["pvalue"] == pytest.approx(1.0)
        assert messages == []

    def test_control_a_group_absent_from_observed_rows_keeps_its_measured_zero(self):
        """0% of 300 observed rows IS a finding, and the severest this scale
        carries. It must not be turned into a could-not-check."""
        df = pd.DataFrame({"gender": ["m"] * 300})

        result, _messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        assert result["measurement_status"] == "measured"
        assert result["groups_not_measured"] == []
        assert result["representation_ratios"]["f"] == 0.0
        assert result["gaps"]["f"] == pytest.approx(-0.5)


# ── 5. calculate_representation_ratio ───────────────────────────────────────


class TestRepresentationRatioTellsALabelMissApartFromAnAbsentGroup:
    """``actual_count == 0`` means either a group genuinely absent from observed
    rows (a finding) or a label that never matched (a could-not-check), and the
    function reported the severest ratio on the scale for both, with no warning.
    """

    def test_a_float_coded_column_no_longer_reports_zero_percent_for_half_the_data(self):
        """Measured on 300 rows of a 1.0 / 0.0 coded column with target_group=1.

        Before: actual_proportion 0.0, representation_ratio 0.0,
        deficit_count 150, is_underrepresented True and no warning, for a group
        holding 150 of those 300 rows.
        """
        df = pd.DataFrame({"sex": np.array([1.0] * 150 + [0.0] * 150)})

        result, messages = _caught(calculate_representation_ratio, df, "sex", 1, 0.5)

        assert result["actual_proportion"] is None
        assert result["representation_ratio"] is None
        assert result["deficit_count"] is None
        assert result["is_underrepresented"] is None
        assert result["measurement_status"] == "could_not_measure"
        assert result["label_status"] == "did_not_match_any_level"
        assert result["observed_levels"] == ["0.0", "1.0"]
        assert _said(messages, "matches 150 of 300 row(s)")
        # And the float spelling of the same label still finds the group.
        assert calculate_representation_ratio(df, "sex", 1.0, 0.5)["actual_count"] == 150

    def test_a_sample_below_min_group_size_is_unassessed_not_underrepresented(self):
        """``_analyze_single_attribute`` in this module refuses anything below
        min_group_size (30) with "UNASSESSED, not adequate". This entry point
        graded ONE row: representation_ratio 0.0, is_underrepresented True, no
        warning.
        """
        result, messages = _caught(
            calculate_representation_ratio, pd.DataFrame({"gender": ["m"]}), "gender", "f", 0.5
        )

        assert result["representation_ratio"] is None
        assert result["is_underrepresented"] is None
        assert result["measurement_status"] == "could_not_measure"
        assert "min_group_size=30" in result["not_measured_reason"]
        assert _said(messages, "UNASSESSED")

    def test_an_infinite_benchmark_is_refused_rather_than_raising(self):
        """Before: OverflowError: cannot convert float infinity to integer, at
        ``int((benchmark_proportion - actual_proportion) * n_total)``, from a
        function whose docstring promises None for a benchmark it cannot divide
        by. ``float('inf') > 0`` is True."""
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        result, messages = _caught(calculate_representation_ratio, df, "gender", "f", float("inf"))

        assert result["representation_ratio"] is None
        assert result["deficit_count"] is None
        assert result["is_underrepresented"] is None
        assert result["measurement_status"] == "could_not_measure"
        assert _said(messages, "not positive and finite")

    def test_a_genuinely_absent_group_keeps_its_measured_zero_and_says_so(self):
        """The distinction the fix exists to keep: no row of 300 carries 'f' and
        no type-tolerant match exists either, so 0.0 is a MEASURED zero. It is
        reported, and now disclosed rather than silent."""
        df = pd.DataFrame({"gender": ["m"] * 300})

        result, messages = _caught(calculate_representation_ratio, df, "gender", "f", 0.5)

        assert result["representation_ratio"] == 0.0
        assert result["is_underrepresented"] is True
        assert result["measurement_status"] == "measured"
        assert result["label_status"] == "absent_from_observed_rows"
        assert _said(messages, "MEASURED zero")

    # ── over-correction control ──

    def test_control_a_real_column_still_gets_its_real_ratio_silently(self):
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        result, messages = _caught(calculate_representation_ratio, df, "gender", "f", 0.5)

        assert result["actual_count"] == 150
        assert result["actual_proportion"] == pytest.approx(0.5)
        assert result["representation_ratio"] == pytest.approx(1.0)
        assert result["deficit_count"] == 0
        assert result["is_underrepresented"] is False
        assert result["measurement_status"] == "measured"
        assert result["label_status"] == "matched"
        assert messages == []


# ── 6. paired_metric_significance ───────────────────────────────────────────


def _single_row_group_inputs():
    rng = np.random.default_rng(11)
    n = 200
    s = np.array(["A"] * (n - 1) + ["B"])
    y = rng.integers(0, 2, n)
    before = rng.integers(0, 2, n)
    after = np.where(s == "B", 1, rng.integers(0, 2, n))
    return y, before, s, after


class TestSignificanceRefusalIsGatedOnTheDeltaNotOnTheReplicates:
    """The guard was ``n_measured >= 2`` alone, so it needed EVERY replicate to
    fail. A protected group of a single row fails the point estimate while the
    replicates that draw that row twice succeed, so 67 of 200 finite replicates
    were enough to publish a verdict about a NaN delta."""

    def test_a_nan_delta_gets_no_p_value_and_no_verdict(self):
        """Measured on 200 rows with group 'B' holding exactly one of them.

        Before: p_value 1.0, significant False, n_bootstrap_measured 67, and the
        prose "... was NOT ASSESSED ... This change is not statistically
        significant (p = 1.0000); it is consistent with sampling noise."
        """
        y, before, s, after = _single_row_group_inputs()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = paired_metric_significance(y, before, s, y, after, s, n_bootstrap=200)
        messages = [str(w.message) for w in caught]

        assert not np.isfinite(result["delta"])
        assert not np.isfinite(result["p_value"])
        assert result["significant"] is None
        assert not np.isfinite(result["delta_ci"][0])
        # The replicate counts are still reported, honestly: the point is that
        # 67 finite replicates no longer buy a verdict.
        assert result["n_bootstrap_measured"] == 67
        assert result["n_bootstrap_unmeasurable"] == 133
        assert "NOT ASSESSED" in result["interpretation"]
        assert "consistent with sampling noise" not in result["interpretation"]
        assert "the delta itself could not be computed" in result["interpretation"]
        assert _said(messages, "although 67 of 200 bootstrap")

    def test_one_protected_group_keeps_its_own_reason(self):
        """The original refusal path, where no replicate is finite either, must
        still name the replicate count rather than the new reason only."""
        rng = np.random.default_rng(3)
        n = 60
        s = np.array(["A"] * n)
        y = rng.integers(0, 2, n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = paired_metric_significance(
                y, rng.integers(0, 2, n), s, y, rng.integers(0, 2, n), s, n_bootstrap=50
            )
        messages = [str(w.message) for w in caught]

        assert not np.isfinite(result["p_value"])
        assert result["significant"] is None
        assert result["n_bootstrap_measured"] == 0
        assert _said(messages, "paired_metric_significance")

    # ── over-correction control ──

    def test_control_a_real_improvement_is_still_measured_as_significant(self):
        rng = np.random.default_rng(7)
        n = 300
        s = rng.choice(["A", "B"], n)
        y = rng.integers(0, 2, n)
        before = (rng.random(n) < np.where(s == "A", 0.8, 0.3)).astype(int)
        after = (rng.random(n) < np.where(s == "A", 0.55, 0.5)).astype(int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = paired_metric_significance(y, before, s, y, after, s, n_bootstrap=200)

        assert result["delta"] == pytest.approx(-0.4325, abs=5e-4)
        assert result["p_value"] == 0.0
        assert result["significant"] is True
        assert "improved (fairer)" in result["interpretation"]
        assert "NOT ASSESSED" not in result["interpretation"]
        assert [str(w.message) for w in caught] == []


# ── 7. plot_correlation_bars ────────────────────────────────────────────────


class TestCorrelationBarsDoesNotDeleteWhatItCouldNotMeasure:
    """``correlations[attr].abs().dropna()`` deleted the unmeasured features, so
    a chart titled "Feature Correlations with <attribute>" showed a feature
    simply missing, which reads as that feature carrying no proxy risk."""

    def test_an_unmeasured_feature_keeps_a_row_and_gets_no_bar(self):
        """Measured on 300 rows: income eta 0.999166, age constant so NaN.

        Before: bars 1, ylabels ['income'], texts [], warnings []. The artists
        are read here, not the ticks: an empty axes still carries a '0.0' tick.
        """
        df = _proxy_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])

        axes, messages = _caught(plot_correlation_bars, matrix, "gender")

        bars = [p for p in axes.patches if p.get_width() > 0]
        assert len(bars) == 1
        assert bars[0].get_width() == pytest.approx(0.9992, abs=1e-3)
        drawn_text = [t.get_text() for t in axes.texts]
        assert "not measured (could not check)" in drawn_text
        assert "age (not measured)" in [t.get_text() for t in axes.get_yticklabels()]
        assert _said(messages, "could not be measured on this data and have NO bar")
        matplotlib.pyplot.close("all")

    def test_a_column_with_nothing_measured_draws_no_chart_at_all(self):
        """Measured on a 5-row frame (below MIN_SAMPLE_SIZE=10) whose whole
        gender column is NaN. Before: an axes with 0 bars, 0 texts, no warning
        and a full title and legend, which reads as no correlation found.
        """
        df = pd.DataFrame(
            {
                "gender": ["m", "f", "m", "f", "m"],
                "income": [1.0, 2.0, 3.0, 4.0, 5.0],
                "age": [20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])

        axes, messages = _caught(plot_correlation_bars, matrix, "gender")

        assert axes is None
        assert _said(messages, "not one feature correlation")
        matplotlib.pyplot.close("all")

    # ── over-correction control ──

    def test_control_a_measurable_frame_still_draws_every_bar_silently(self):
        rng = np.random.default_rng(4)
        n = 400
        g = rng.integers(0, 2, n)
        df = pd.DataFrame(
            {
                "g": np.where(g == 1, "m", "f"),
                "age": g * 10.0 + rng.normal(30, 3, n),
                "inc": rng.normal(50, 8, n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["g"], feature_columns=["age", "inc"])

        axes, messages = _caught(plot_correlation_bars, matrix, "g")

        bars = [p for p in axes.patches if p.get_width() > 0]
        assert len(bars) == 2
        assert max(b.get_width() for b in bars) > 0.5, "the real association lost its bar"
        assert [t.get_text() for t in axes.texts] == []
        assert messages == []
        matplotlib.pyplot.close("all")


# ── 8. plot_intersectional_heatmap ──────────────────────────────────────────


class TestIntersectionalHeatmapReadsItsOwnCoverageRecord:
    """The function read neither ``coverage`` nor ``groups_not_assessed`` from a
    result carrying both, and no test in the repository executed it at all."""

    def test_an_unexamined_cell_keeps_a_slot_and_gets_no_bar(self):
        """Measured with one intersectional cell shrunk to 3 rows against
        min_group_size 30: the result says coverage 'partial' and names 'f_b'.

        Before: bars 3, xlabels ['f_a', 'm_b', 'm_a'], texts [], warnings [],
        under the title 'Mean "score" Across Intersectional Groups'.
        """
        df = _intersectional_frame()
        shrink = (df["g"] == "f") & (df["r"] == "b")
        df = df.drop(index=df.index[shrink][3:])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyze_intersectional_correlations(df, ["g", "r"])
        assert results["coverage"] == "partial"
        assert [e["group"] for e in results["groups_not_assessed"]] == ["f_b"]

        axes, messages = _caught(plot_intersectional_heatmap, results, "score")

        bars = [p for p in axes.patches if p.get_height() != 0]
        assert len(bars) == 3
        assert "not examined" in [t.get_text() for t in axes.texts]
        assert "f_b (not examined)" in [t.get_text() for t in axes.get_xticklabels()]
        assert _said(messages, "never examined and have NO bar")
        matplotlib.pyplot.close("all")

    def test_no_group_examined_says_so_instead_of_blaming_the_feature_name(self):
        """Before: it returned None warning "Feature 'score' not found in
        intersectional results", which sends the caller to fix a naming problem
        that does not exist: the feature IS in the frame and the real reason is
        that no group reached the minimum."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyze_intersectional_correlations(
                _intersectional_frame().head(20), ["g", "r"]
            )
        assert results["coverage"] == "not_assessed"

        axes, messages = _caught(plot_intersectional_heatmap, results, "score")

        assert axes is None
        assert _said(messages, "not one of the 4 intersectional group(s) was examined")
        assert not _said(messages, "not found in intersectional results")
        matplotlib.pyplot.close("all")

    def test_a_feature_that_really_is_absent_still_says_not_found(self):
        """The other direction: with every group examined, a feature name that
        is not in the results IS a naming problem, and the message stays."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyze_intersectional_correlations(_intersectional_frame(), ["g", "r"])

        axes, messages = _caught(plot_intersectional_heatmap, results, "no_such_feature")

        assert axes is None
        assert _said(messages, "not found in intersectional results")
        matplotlib.pyplot.close("all")

    # ── over-correction control ──

    def test_control_complete_coverage_draws_every_cell_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyze_intersectional_correlations(_intersectional_frame(), ["g", "r"])
        assert results["coverage"] == "complete"

        axes, messages = _caught(plot_intersectional_heatmap, results, "score")

        bars = [p for p in axes.patches if p.get_height() != 0]
        assert len(bars) == 4
        assert all(45.0 < b.get_height() < 55.0 for b in bars), "the group means lost their value"
        assert [t.get_text() for t in axes.texts] == []
        assert messages == []
        matplotlib.pyplot.close("all")


# ── 9. create_analysis_dashboard ────────────────────────────────────────────


def _panel_text(fig) -> str:
    return "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)


class TestDashboardDoesNotPublishTheGraderFallThroughBand:
    """``identify_proxy_variables`` marks a pair whose headline association could
    not be measured in three places, and the dashboard read two of them. The one
    it dropped is the one that turned an unmeasurable pair into a NEGLIGIBLE risk
    slice and a Top Concerns entry."""

    def test_an_unmeasured_pair_is_named_not_graded(self):
        """Measured on 300 rows: income determines gender, age is constant.

        Before: 'CRITICAL (1) 50.0% / NEGLIGIBLE (1) 50.0%' in the pie, 'nan' as
        a bar value, 'Top Concerns: income (critical), age (negligible)', beside
        the panel's own sentence "Their absence below is a could-not-check"
        about a pair that was present below.
        """
        df = _proxy_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            proxies = identify_proxy_variables(df, ["gender"])
            fig = create_analysis_dashboard(matrix, proxies)

        age = [p for p in proxies if p.feature == "age"][0]
        assert not np.isfinite(age.correlation)
        assert age.evidence["correlation_measures"]["headline_measure_status"] == "not_assessed"

        text = _panel_text(fig)
        assert "NOT GRADED: 1 screened pair(s)" in text
        assert "gender ~ age" in text
        assert "NEGLIGIBLE" not in text
        assert "(negligible)" not in text
        assert "nan" not in text
        # The measured half is untouched and the coverage sentence is now true.
        assert "CRITICAL: 1" in text
        assert "NOT MEASURED: 1 of 2" in text
        matplotlib.pyplot.close("all")

    def test_a_frame_with_no_feature_pair_does_not_read_as_clean(self):
        """A frame whose only column is the protected attribute: 0 of 0 cells.
        Before: 'Features analyzed: 0 of 0' beside the unqualified 'none found'.
        """
        df = pd.DataFrame({"gender": ["m", "f"] * 50})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            fig = create_analysis_dashboard(matrix, [])

        text = _panel_text(fig)
        assert "none found, nothing was examined" in text
        matplotlib.pyplot.close("all")

    # ── over-correction control ──

    def test_control_a_measured_dashboard_still_grades_its_finding(self):
        rng = np.random.default_rng(0)
        n = 300
        g = rng.integers(0, 2, n)
        df = pd.DataFrame(
            {"gender": np.where(g == 1, "m", "f"), "income": g * 50.0 + rng.normal(0, 1, n)}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            proxies = identify_proxy_variables(df, ["gender"])
            fig = create_analysis_dashboard(matrix, proxies)

        text = _panel_text(fig)
        assert "Features analyzed: 1 of 1" in text
        assert "CRITICAL: 1" in text
        assert "income" in text
        assert "NOT GRADED" not in text
        assert "NOT MEASURED" not in text
        matplotlib.pyplot.close("all")


# ── 10. prepare_protected_attributes ────────────────────────────────────────


class TestSingleGroupDisclosureSitsAboveTheBinningDispatch:
    """The Coverage note lived inside branch 4 (low-cardinality, used as-is), so
    it could only describe a column that ARRIVED with one level. Branches 1, 2,
    4b, 6 and 8 reach ``usable.append`` without it, and each of them bins, which
    is how a single group gets manufactured out of a multi-level column."""

    def test_zip_codes_from_one_area_are_disclosed_as_a_single_group(self):
        """Measured on 400 rows over 60 distinct ZIPs all in the 100xx area:
        the rolled-up column has exactly one level, 'ZIP 100xx'.

        Before: usable ['zipcode'] with a Binning note only and no Coverage note.
        """
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"zipcode": rng.choice([f"100{i:02d}" for i in range(60)], 400)})

        prep = prepare_protected_attributes(df, ["zipcode"])

        assert prep.usable == ["zipcode"], "excluding it would be the over-correction"
        assert prep.frame["zipcode"].nunique(dropna=True) == 1
        labels = [label for label, _detail in prep.notes]
        assert labels == ["Binning", "Coverage"]
        assert any("single group" in detail for _label, detail in prep.notes)

    def test_an_age_column_banded_into_one_band_is_disclosed_too(self):
        """300 rows whose ages all fall inside 25-34: 20 distinct raw values, so
        branch 2 bands it and the banded column has one level, '25-34'."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"age": rng.choice(np.arange(25.0, 34.99, 0.5), 300)})

        prep = prepare_protected_attributes(df, ["age"])

        assert prep.usable == ["age"]
        assert prep.frame["age"].nunique(dropna=True) == 1
        assert "Coverage" in [label for label, _detail in prep.notes]

    def test_a_raw_single_level_column_is_still_disclosed(self):
        """The case the note was originally written for must survive the move."""
        prep = prepare_protected_attributes(pd.DataFrame({"gender": ["f"] * 100}), ["gender"])

        assert prep.usable == ["gender"]
        assert prep.excluded == []
        assert [label for label, _detail in prep.notes] == ["Coverage"]

    # ── over-correction controls ──

    def test_control_a_real_two_level_attribute_discloses_nothing(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"gender": rng.choice(["m", "f"], 100)})

        prep = prepare_protected_attributes(df, ["gender"])

        assert prep.usable == ["gender"]
        assert prep.excluded == []
        assert prep.notes == []

    def test_control_binning_that_keeps_several_groups_gets_no_coverage_note(self):
        """The same two binning branches on inputs that really do yield groups:
        60 ZIPs across three areas, and ages spread over the real bands."""
        rng = np.random.default_rng(1)
        zips = [f"{a}00{i:02d}" for a in (1, 2, 3) for i in range(20)]
        prep_zip = prepare_protected_attributes(
            pd.DataFrame({"zipcode": rng.choice(zips, 400)}), ["zipcode"]
        )
        assert prep_zip.frame["zipcode"].nunique(dropna=True) == 3
        assert [label for label, _detail in prep_zip.notes] == ["Binning"]

        prep_age = prepare_protected_attributes(
            pd.DataFrame({"age": rng.integers(18, 80, 300)}), ["age"]
        )
        assert prep_age.frame["age"].nunique(dropna=True) == 6
        assert [label for label, _detail in prep_age.notes] == ["Binning"]
