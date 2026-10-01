"""BGL4 audit of batch A-preprocessing-3: the overturns, INVERTED once fixed.

READ THIS BEFORE EDITING.

Every test below was written to assert the DEFECTIVE output the code produced on
2026-09-27, so each passing test WAS a recorded defect. All ten overturns are now
closed in BGL5, so every assertion here has been INVERTED to the corrected
behaviour: same subject, same fixture, same docstring, opposite expectation. The
"Measured:" numbers in the docstrings are kept verbatim as the record of what the
defect looked like, and each test now says what the value is instead.

Inverted 2026-09-27 (22 test items across 13 functions). The pins with the
over-correction controls beside them live in tests/test_bgl5_preprocessing_3.py;
this file is the before/after record, and it is deliberately redundant with that
one so a regression reddens both.

The two remaining overturns (``plot_correlation_bars`` and
``plot_intersectional_heatmap``) are figure-shaped and are pinned only in the BGL5
file, by reading the drawn artists.
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
    compute_feature_correlations,
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


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


def _zip_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    zips = rng.choice([f"{z:05d}" for z in range(10001, 10061)], n)
    inner = np.isin(zips, [f"{z:05d}" for z in range(10001, 10031)])
    approved = np.where(inner, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(int)
    return pd.DataFrame({"zip": zips, "approved": approved})


class TestRedliningCoverageIsOnlyThreeStateWhenEVERYUnitIsNull:
    """OVERTURNED ``check_geographic_redlining_risk`` PROVEN. CLOSED in BGL5.

    The fix decided on ``np.isfinite(outcome_range)``, and a range is finite as
    soon as ONE geographic unit carries an observation, because
    ``Series.max()`` and ``Series.min()`` skip NaN. So the refusal reached only
    the all-null column the pin tried. It now needs TWO units with a finite mean,
    because a range is a comparison.
    """

    def test_one_observed_zip_of_sixty_reports_complete_coverage_and_a_score_of_zero(self):
        """Measured: 13 of 600 rows observed, in 1 of 60 ZIPs.

        Honest output would be ``assessment_coverage 'partial'`` with
        ``outcome_disparity`` named in ``components_not_assessed``: a range
        taken over a single unit is not a between-unit comparison, and 0.0 is
        the strongest all-clear this score can give. That is now what it returns.
        """
        df = _zip_frame()
        keep = df["zip"] == df["zip"].iloc[0]
        df.loc[~keep, "approved"] = np.nan
        assert int(df["approved"].notna().sum()) == 13
        assert df.loc[df["approved"].notna(), "zip"].nunique() == 1
        assert df["zip"].nunique() == 60

        result, messages = _caught(
            check_geographic_redlining_risk, df, "zip", outcome_column="approved"
        )

        # INVERTED: the four defective values, one by one.
        assert result["assessment_coverage"] == "partial"
        assert [e["component"] for e in result["components_not_assessed"]] == ["outcome_disparity"]
        assert "finite over 1 of 60" in result["components_not_assessed"][0]["reason"]
        assert result["outcome_analysis"]["outcome_range"] is None
        assert result["outcome_analysis"]["n_geographic_units"] == 60
        assert result["outcome_analysis"]["n_geographic_units_with_a_finite_mean"] == 1
        assert any("LOWER BOUND" in m for m in messages)
        # The tell that used to be the ONLY one, and was buried inside
        # outcome_analysis: a std over a single unit is NaN. It is still NaN, and
        # the top level now says so too.
        assert not np.isfinite(result["outcome_analysis"]["mean_by_geo_std"])


class TestRepresentationRatioHasNoThirdStateForALabelThatMatchesNothing:
    """OVERTURNED ``calculate_representation_ratio`` PROVEN. CLOSED in BGL5.

    ``actual_count == 0`` carries two different meanings and the function reports
    the same number for both: the group is genuinely absent from the observed
    rows (a finding), or the label never matched this column at all (a
    could-not-check). The pin covered only ``n_total <= 0``.
    """

    def test_a_float_coded_column_reports_zero_percent_for_half_the_dataset(self):
        """Measured on 300 rows of a 1.0 / 0.0 coded column, target_group=1.

        ``series.astype(str)`` gives '1.0' and ``_group_key(1)`` gives '1', so
        nothing matches and the function reports ``representation_ratio 0.0``
        with ``is_underrepresented True`` for a group holding 150 of 300 rows,
        with no warning. Honest output would be None with a warning naming the
        levels the column actually holds, and that is now what it returns.
        """
        df = pd.DataFrame({"sex": np.array([1.0] * 150 + [0.0] * 150)})

        result, messages = _caught(calculate_representation_ratio, df, "sex", 1, 0.5)

        # INVERTED. actual_count stays 0 because that IS what the comparison
        # counted; every graded field derived from it is now None.
        assert result["actual_count"] == 0
        assert result["actual_proportion"] is None
        assert result["representation_ratio"] is None
        assert result["is_underrepresented"] is None
        assert result["label_status"] == "did_not_match_any_level"
        assert result["observed_levels"] == ["0.0", "1.0"]
        assert any("matches 150 of 300 row(s)" in m for m in messages)
        # The same call with the float spelling finds the group: the data is not
        # what is wrong here.
        assert calculate_representation_ratio(df, "sex", 1.0, 0.5)["actual_count"] == 150

    def test_a_single_row_gets_a_graded_verdict_the_module_itself_refuses(self):
        """``_analyze_single_attribute`` in this same module refuses anything
        below ``min_group_size`` (default 30) with "UNASSESSED, not adequate",
        and ``RepresentationSeverity.INSUFFICIENT_DATA`` exists for exactly
        this. This entry point graded one row with no disclosure at all; it now
        applies the same gate, exposed as a ``min_group_size`` argument.
        """
        result, messages = _caught(
            calculate_representation_ratio, pd.DataFrame({"gender": ["m"]}), "gender", "f", 0.5
        )

        # INVERTED.
        assert result["representation_ratio"] is None
        assert result["is_underrepresented"] is None
        assert result["measurement_status"] == "could_not_measure"
        assert "min_group_size=30" in result["not_measured_reason"]
        assert any("UNASSESSED" in m for m in messages)

    def test_a_positive_infinite_benchmark_crashes_instead_of_refusing(self):
        """The docstring promises None for a benchmark that cannot be divided by.
        ``float('inf') > 0`` was True, so it reached ``int(inf * n)`` and raised
        OverflowError. The test is now ``_is_a_divisible_share``, so it refuses.
        """
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        # INVERTED: no exception, and the None the docstring always promised.
        result, messages = _caught(calculate_representation_ratio, df, "gender", "f", float("inf"))

        assert result["representation_ratio"] is None
        assert result["deficit_count"] is None
        assert result["is_underrepresented"] is None
        assert any("not positive and finite" in m for m in messages)


class TestBenchmarkGroupsNotMeasuredIsEmptyWheneverAnyRowWasObserved:
    """OVERTURNED ``compare_to_benchmark`` PROVEN. CLOSED in BGL5.

    The unit's own comment promises "the groups no ratio was reported for named
    either way". The implementation was ``[] if measured else groups_seen``, and
    ``measured`` was ``n_total > 0``, so a group whose ratio genuinely could not
    be divided was dropped from both dicts while the result declared that
    nothing went unmeasured. The list is now built where the ratios are, one
    entry per group that got none, and ``measurement_status`` has a "partial".
    """

    @pytest.mark.parametrize("expected", [0.0, -0.1, float("nan")])
    def test_a_group_with_no_positive_expected_share_is_dropped_silently(self, expected):
        """Measured on 300 rows against {'m': 0.5, 'f': 0.5, 'x': <expected>}.

        Honest output would name 'x' in ``groups_not_measured``, because
        ``actual_prop / expected_prop`` does not exist for it. That is now what
        it returns.
        """
        df = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})

        result, messages = _caught(
            compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5, "x": expected}
        )

        # INVERTED. 'x' is still absent from both dicts, which is correct: there
        # is no ratio to put there. What changed is that it is now NAMED.
        assert "x" not in dict(result["representation_ratios"])
        assert "x" not in dict(result["gaps"])
        assert result["measurement_status"] == "partial"
        assert result["groups_not_measured"] == ["x"]
        assert any("NOT a ratio of 0" in m for m in messages)

    def test_an_observed_group_the_benchmark_never_mentions_is_dropped_the_same_way(self):
        """20 of 300 rows are 'nb', which the benchmark does not mention. Its
        ratio is not reported and ``groups_not_measured`` stays empty. The
        chi-squared half DOES disclose it, which is what makes the ratio half's
        silence visible as an inconsistency rather than a policy. The two halves
        now agree.
        """
        df = pd.DataFrame({"gender": ["m"] * 140 + ["f"] * 140 + ["nb"] * 20})

        result, messages = _caught(compare_to_benchmark, df, "gender", {"m": 0.5, "f": 0.5})

        # INVERTED.
        assert "nb" not in dict(result["representation_ratios"])
        assert result["groups_not_measured"] == ["nb"]
        assert result["measurement_status"] == "partial"
        assert result["pvalue"] is None
        assert any("the benchmark never" in m for m in messages)


class TestSignificanceRefusalNeedsEVERYReplicateToBeUnmeasurable:
    """OVERTURNED ``paired_metric_significance`` PROVEN. CLOSED in BGL5.

    The guard was ``n_measured >= 2``. A protected group of a SINGLE row makes
    the point estimate unmeasurable (it is below ``min_group_size``) while
    bootstrap replicates that draw that row twice are measurable, so 67 of 200
    replicates were finite and the guard never fired. The guard now also requires
    the point-estimate delta to be finite, because that is what the p-value is
    about.
    """

    def test_a_nan_delta_is_published_with_p_one_and_significant_false(self):
        """Measured on 200 rows, group 'B' holding exactly one of them.

        This is byte-identical to the signature the fix's own comment condemns:
        "value_before=nan, value_after=nan, delta=nan, p_value=1.0,
        significant=False, which is the signature of a test that ran and found
        nothing". Honest output would be p nan and significant None, as the
        docstring promises, because a p-value about a NaN delta is a p-value
        about nothing. That is now what it returns.
        """
        rng = np.random.default_rng(11)
        n = 200
        s = np.array(["A"] * (n - 1) + ["B"])
        y = rng.integers(0, 2, n)
        before = rng.integers(0, 2, n)
        after = np.where(s == "B", 1, rng.integers(0, 2, n))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = paired_metric_significance(y, before, s, y, after, s, n_bootstrap=200)

        assert not np.isfinite(result["value_before"])
        assert not np.isfinite(result["value_after"])
        assert not np.isfinite(result["delta"])
        # INVERTED: no verdict about a change that does not exist.
        assert not np.isfinite(result["p_value"])
        assert result["significant"] is None
        # The replicate counts are unchanged and still reported: 67 finite
        # replicates simply no longer buy a verdict.
        assert result["n_bootstrap_measured"] == 67
        assert result["n_bootstrap_unmeasurable"] == 133
        # And the prose no longer contradicts itself inside one sentence.
        assert "NOT ASSESSED" in result["interpretation"]
        assert "not statistically significant (p = 1.0000)" not in result["interpretation"]
        assert "consistent with sampling noise" not in result["interpretation"]
        assert "the delta itself could not be computed" in result["interpretation"]


class TestDashboardPublishesTheGraderFallThroughBandAsAMeasuredRisk:
    """OVERTURNED ``create_analysis_dashboard`` PROVEN. CLOSED in BGL5.

    ``identify_proxy_variables`` records, per result,
    ``evidence['correlation_measures']['headline_measure_status'] ==
    'not_assessed'`` plus a recommendation saying in so many words that
    "negligible" is "the grader's fall-through band for a value that is not a
    number; it is NOT a finding of low risk". The dashboard read neither, and
    published that band as a risk-distribution slice and a Top Concerns entry. It
    now keeps such a pair out of both and names it under NOT GRADED.
    """

    def test_an_unmeasured_pair_is_drawn_as_a_negligible_risk_beside_the_not_measured_line(self):
        """Measured on 300 rows: 'income' determines 'gender', 'age' is constant
        so its association with 'gender' cannot be computed.
        """
        rng = np.random.default_rng(0)
        n = 300
        g = rng.integers(0, 2, n)
        df = pd.DataFrame(
            {
                "gender": np.where(g == 1, "m", "f"),
                "income": g * 50.0 + rng.normal(0, 1, n),
                "age": np.full(n, 40.0),
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_feature_correlations(df, ["gender"])
            proxies = identify_proxy_variables(df, ["gender"])
            fig = create_analysis_dashboard(matrix, proxies)

        age = [p for p in proxies if p.feature == "age"][0]
        # The producer knows, and says so twice.
        assert not np.isfinite(age.correlation)
        assert age.evidence["correlation_measures"]["headline_measure_status"] == "not_assessed"
        assert "NOT ASSESSED" in age.recommendations[0]
        assert age.risk_level.value == "negligible"

        text = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
        # INVERTED: the picture no longer grades it, so the panel's own sentence
        # about the pair being ABSENT below is now true.
        assert "NEGLIGIBLE" not in text
        assert "(negligible)" not in text
        assert "NOT GRADED: 1 screened pair(s)" in text
        assert "gender ~ age" in text
        assert "NOT MEASURED: 1 of 2" in text
        assert "absence below is a could-not-check" in text
        # The measured half survives: the fix is not a blanket refusal.
        assert "CRITICAL: 1" in text
        matplotlib.pyplot.close("all")


class TestSingleGroupDisclosureIsBelowTheBinningDispatch:
    """OVERTURNED ``prepare_protected_attributes`` PROVEN. CLOSED in BGL5.

    The ``nunique == 1`` Coverage note sat inside branch 4 (low-cardinality,
    used as-is). Branches 1, 2, 4b, 6 and 8 reach ``usable.append(col)``
    without it, and their own binning can MANUFACTURE a single group out of a
    multi-level column, which is the case the note exists for. The note now runs
    once, ABOVE the dispatch, over the post-binning column.
    """

    def test_zip_codes_from_one_area_become_one_group_with_no_coverage_note(self):
        """Measured on 400 rows over 60 distinct ZIPs, all in the 100xx area:
        the rolled-up column has exactly one level and is returned usable with a
        Binning note only. Honest output would add the same Coverage note the
        raw single-level branch adds, and that is now what it returns.
        """
        rng = np.random.default_rng(1)
        zips = [f"100{i:02d}" for i in range(60)]
        df = pd.DataFrame({"zipcode": rng.choice(zips, 400)})

        prep = prepare_protected_attributes(df, ["zipcode"])

        # INVERTED. Still usable (excluding it would be the over-correction) and
        # still binned; the Coverage note is what was missing.
        assert prep.usable == ["zipcode"]
        assert prep.frame["zipcode"].nunique(dropna=True) == 1
        assert [label for label, _detail in prep.notes] == ["Binning", "Coverage"]
        assert any("single group" in detail for _label, detail in prep.notes)

    def test_an_age_column_banded_into_one_band_gets_no_coverage_note_either(self):
        """300 rows whose ages all fall inside 25-34: 20 distinct raw values, so
        branch 2 bands it, and the banded column has one level.
        """
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"age": rng.choice(np.arange(25.0, 34.99, 0.5), 300)})

        prep = prepare_protected_attributes(df, ["age"])

        # INVERTED.
        assert prep.usable == ["age"]
        assert prep.frame["age"].nunique(dropna=True) == 1
        assert [label for label, _detail in prep.notes] == ["Binning", "Coverage"]


class TestHistoricalPrecedentIsAttachedBySUBSTRINGMatch:
    """OVERTURNED ``attribute_historical_pattern`` and
    ``domain_historical_context`` SEMI-PROVEN. CLOSED in BGL5.

    Both docstrings promise silence for an unknown input ("silence beats
    invention", "no fabrication"). Both resolved by ``alias in key`` over a raw
    lowercased string, so an unrelated name collided with a curated precedent
    and came back carrying REAL citations, which is the form a reader cannot
    audit. ``protected_binning``'s module docstring records this exact bug class
    ('"age" matching "primary_language"') as a past incident, and
    ``vfairness._names.name_tokens`` exists as the whole-token fix. Both lookups
    now route through it, longest match first.
    """

    @pytest.mark.parametrize(
        "column", ["primary_language", "average_salary", "message_count", "percentage_complete"]
    )
    def test_any_name_containing_the_letters_age_gets_the_adea_precedent(self, column):
        # INVERTED: the honest answer for this column is None, and it is now None.
        assert attribute_historical_pattern(column, "hiring") is None

    def test_a_contract_column_gets_the_lending_redlining_precedent(self):
        """'contract_type' contains 'tract'."""
        # INVERTED: 'tract' is not a whole token of 'contract_type'.
        assert attribute_historical_pattern("contract_type", "lending") is None

    @pytest.mark.parametrize(
        "domain", ["chronic disease management", "threat detection", "anthropology research"]
    )
    def test_any_domain_containing_hr_gets_the_hiring_precedent(self, domain):
        """The alias 'hr' resolved to 'hiring' by substring, so a chronic-disease
        domain was handed the Amazon-recruiting-tool precedent.
        """
        # INVERTED: 'hr' is not a whole token of 'chronic', 'threat' or
        # 'anthropology', so the honest answer for all three is None.
        assert domain_historical_context(domain) is None

    @pytest.mark.parametrize("domain", ["bailout underwriting", "mortgage bail-in"])
    def test_a_bailout_domain_gets_the_recidivism_precedent(self, domain):
        """The alias 'bail' resolved to 'justice' by substring."""
        out = domain_historical_context(domain)

        # INVERTED, and the two inputs part company, which is the point: 'bail' is
        # not a token of 'bailout' at all, while it IS a token of 'bail-in', where
        # the longer whole-token match 'mortgage' decides and gives lending.
        assert (out or {}).get("domain") != "justice"
        if domain == "bailout underwriting":
            assert out is None
        else:
            assert out is not None and out["domain"] == "lending"
