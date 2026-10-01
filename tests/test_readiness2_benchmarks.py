"""Readiness-2, benchmarks lane: a perfectly representative dataset reported CRITICAL.

THE DEFECT. ``DEFAULT_BENCHMARKS`` was a spelling-ALIAS table that the code
consumed as a probability distribution:

    "gender": {"male": 0.49, "female": 0.51, "m": 0.49, "f": 0.51,
               "man": 0.49, "woman": 0.51}

Measured from a clean install, on 4900 male + 5100 female rows, which is the
built-in benchmark to the last row:

    detect_representation_bias(df, ["gender"])
      severity          : CRITICAL
      ratios            : female 3.0, male 3.0, m 0.0, f 0.0, man 0.0, woman 0.0
      underrepresented  : m, f, man, woman (absent=True, count=0), each with an
                          expected_proportion of 0.1633 and a deficit of 1633
      chi_squared       : None / None
      recommendations   : "CRITICAL: Severe underrepresentation detected",
                          "Underrepresented groups requiring attention: m, f, man"

Four faults compounded, and they are pinned separately below because any one of
them can come back on its own:

  a. aliases carried probability mass, so gender summed to 3.000 (race 1.523,
     age_group 2.170);
  b. the renormalisation divided by that sum, turning 0.49 into 0.1633;
  c. the absent-group loop reports any benchmark key missing from the data as a
     missing demographic group, so spelling variants became phantom populations
     with deficits in the thousands;
  d. scipy refused the mismatched totals and a bare ``except Exception: return
     None, None`` swallowed it, so CRITICAL was emitted with no statistic.

Blast radius: ``vfairness.detect_representation_bias`` and
``vfairness.BiasDetector(...).full_audit`` are both in ``vfairness.__all__``.
full_audit reported overall_risk_score 0.5 and a critical issue naming m, f and
man.

SECOND DEFECT, same file. ``detect_representation_bias(significance_level=...)``
was INERT: documented as the alpha for the chi-squared test, threaded down into
``_analyze_single_attribute``, and never read there. Four different alphas
produced byte-identical results.

Every fix below carries both a refusal pin and an over-correction control that
asserts MEASURED values, so a change that grades everything ADEQUATE fails just
as loudly as the original that graded a clean dataset CRITICAL.
"""

from __future__ import annotations

import warnings

import pytest

pd = pytest.importorskip("pandas")

from vfairness.preprocessing.bias_detection.detector import BiasDetector  # noqa: E402
from vfairness.preprocessing.bias_detection.representation import (  # noqa: E402
    _DEFAULT_BENCHMARK_SCHEMES,
    DEFAULT_BENCHMARK_ALIASES,
    DEFAULT_BENCHMARKS,
    RepresentationSeverity,
    _BenchmarkScheme,
    _validate_default_schemes,
    detect_representation_bias,
)

# The exact dataset the assessment measured: 49/51, the built-in gender
# benchmark, expressed as rows.
REPRESENTATIVE_GENDER = pd.DataFrame({"gender": ["male"] * 4900 + ["female"] * 5100})

# The spellings that used to be graded as demographic groups of their own.
PHANTOM_GROUPS = {"m", "f", "man", "woman", "men", "women"}


def _detect(df, attribute="gender", **kwargs):
    """Call the API without letting an unrelated warning decide the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (result,) = detect_representation_bias(
            df, [attribute], include_intersectional=False, **kwargs
        )
    return result


# ── a: the shipped table must be a distribution, not an alias list ──────────


def test_every_built_in_distribution_sums_to_exactly_one():
    """The measurement that was never taken. gender summed to 3.000."""
    totals = {
        (attribute, scheme.name): sum(scheme.distribution.values())
        for attribute, schemes in _DEFAULT_BENCHMARK_SCHEMES.items()
        for scheme in schemes
    }
    assert totals, "no built-in benchmark schemes are declared at all"
    for key, total in totals.items():
        assert total == pytest.approx(1.0, abs=1e-9), f"{key} sums to {total!r}"

    # And the public view of them, which is what the docstrings point callers at.
    assert sorted(DEFAULT_BENCHMARKS) == ["age_group", "gender", "race"]
    assert DEFAULT_BENCHMARKS["gender"] == {"male": 0.49, "female": 0.51}
    for attribute, table in DEFAULT_BENCHMARKS.items():
        assert sum(table.values()) == pytest.approx(1.0, abs=1e-9), attribute


def test_no_spelling_variant_is_a_category_of_the_distribution():
    """An alias normalises ONTO a category; it never is one."""
    for attribute, schemes in _DEFAULT_BENCHMARK_SCHEMES.items():
        for scheme in schemes:
            categories = set(scheme.distribution)
            assert not (set(scheme.aliases) & categories), (
                f"{scheme.name} declares a label as both a category and an alias"
            )
            assert set(scheme.aliases.values()) <= categories, (
                f"{scheme.name} aliases onto something that is not a category"
            )
        assert not (PHANTOM_GROUPS & set(DEFAULT_BENCHMARKS[attribute]))

    # The gender aliases specifically, since they are the ones that shipped as
    # phantom populations.
    assert DEFAULT_BENCHMARK_ALIASES["gender"] == {
        "m": "male",
        "man": "male",
        "men": "male",
        "f": "female",
        "woman": "female",
        "women": "female",
    }


def test_the_import_time_validator_refuses_the_table_that_shipped():
    """Reinstate the shipped table as data; the validator must refuse it.

    This is the guard that makes the defect unshippable rather than merely
    absent today. The table below is the one measured in production.
    """
    shipped = {
        "gender": (
            _BenchmarkScheme(
                name="the_table_that_shipped",
                distribution={
                    "male": 0.49,
                    "female": 0.51,
                    "m": 0.49,
                    "f": 0.51,
                    "man": 0.49,
                    "woman": 0.51,
                },
                aliases={},
            ),
        )
    }
    assert sum(shipped["gender"][0].distribution.values()) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="not 1.0"):
        _validate_default_schemes(shipped)

    # The other two shapes the validator has to refuse.
    both = {
        "gender": (
            _BenchmarkScheme(
                name="alias_is_also_a_category",
                distribution={"male": 0.49, "female": 0.51},
                aliases={"male": "female"},
            ),
        )
    }
    with pytest.raises(ValueError, match="BOTH a category and an alias"):
        _validate_default_schemes(both)

    dangling = {
        "gender": (
            _BenchmarkScheme(
                name="alias_points_nowhere",
                distribution={"male": 0.49, "female": 0.51},
                aliases={"m": "masculine"},
            ),
        )
    }
    with pytest.raises(ValueError, match="not one of its categories"):
        _validate_default_schemes(dangling)


def test_the_live_table_passes_its_own_validator():
    """Over-correction control: the refusal above is not refusing everything."""
    _validate_default_schemes(_DEFAULT_BENCHMARK_SCHEMES)


# ── b + c: the verdict on a dataset that IS the benchmark ───────────────────


def test_the_representative_dataset_is_adequate_not_critical():
    """The headline. 4900/5100 against the built-in 49/51."""
    result = _detect(REPRESENTATIVE_GENDER)

    assert result.severity == RepresentationSeverity.ADEQUATE
    assert result.benchmark_source == "default_us_census"
    assert dict(result.representation_ratios) == {"female": 1.0, "male": 1.0}
    assert result.underrepresented_groups == []
    assert result.overrepresented_groups == []
    assert result.sample_size == 10000

    # d: the verdict is not graded off a missing statistic any more.
    assert result.chi_squared_statistic == 0.0
    assert result.chi_squared_pvalue == pytest.approx(1.0)
    assert result.chi_squared_significant is False

    joined = " ".join(result.recommendations)
    assert "CRITICAL" not in joined
    assert "Severe underrepresentation" not in joined


def test_a_spelling_variant_is_never_reported_as_a_missing_population():
    """c, in its own right. The deficits were 1633 and 1700 PEOPLE."""
    result = _detect(REPRESENTATIVE_GENDER)

    named = {g["group"] for g in result.underrepresented_groups}
    named |= {g["group"] for g in result.overrepresented_groups}
    named |= set(result.representation_ratios)
    assert not (named & PHANTOM_GROUPS), f"a spelling variant was graded: {named}"
    assert not any(g.get("absent") for g in result.underrepresented_groups)


def test_full_audit_reports_no_critical_issue_on_representative_data():
    """The blast radius, through the other public entry point.

    ``BiasDetector.full_audit`` reported overall_risk_score 0.5 and a critical
    issue reading "Severe underrepresentation in 'gender': ['m', 'f', 'man']".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = BiasDetector(REPRESENTATIVE_GENDER, protected_attributes=["gender"]).full_audit()

    assert report.critical_issues == []
    assert report.overall_risk_score == 0.0
    (finding,) = report.representation_findings
    assert finding.severity == RepresentationSeverity.ADEQUATE


def test_race_and_age_grade_their_own_distributions_adequate():
    """The other two tables, whose sums were 1.523 and 2.170."""
    race = pd.DataFrame(
        {
            "race": ["white"] * 5760
            + ["black"] * 1210
            + ["hispanic"] * 1870
            + ["asian"] * 590
            + ["native_american"] * 70
            + ["pacific_islander"] * 20
            + ["multiracial"] * 280
            + ["other"] * 200
        }
    )
    result = _detect(race, "race")
    assert result.severity == RepresentationSeverity.ADEQUATE
    assert dict(result.representation_ratios) == {k: 1.0 for k in DEFAULT_BENCHMARKS["race"]}
    assert result.chi_squared_statistic == 0.0

    age = pd.DataFrame(
        {
            "age_group": ["under_18"] * 220
            + ["18-24"] * 90
            + ["25-34"] * 140
            + ["35-44"] * 130
            + ["45-54"] * 120
            + ["55-64"] * 130
            + ["65+"] * 170
        }
    )
    detailed = _detect(age, "age_group")
    assert detailed.severity == RepresentationSeverity.ADEQUATE
    assert dict(detailed.representation_ratios) == {k: 1.0 for k in DEFAULT_BENCHMARKS["age_group"]}
    assert detailed.chi_squared_statistic == 0.0


def test_the_coarse_age_banding_is_a_scheme_not_an_alias_set():
    """young / middle / senior are AGGREGATES of the detailed bands.

    They summed to 1.00 on their own inside a table that also held the seven
    detailed bands, which is how age_group reached 2.170. Neither banding is an
    alias of the other, so the one that matches the data's labels is selected.
    """
    coarse = pd.DataFrame({"age_group": ["young"] * 310 + ["middle"] * 390 + ["senior"] * 300})
    result = _detect(coarse, "age_group")
    assert result.severity == RepresentationSeverity.ADEQUATE
    assert dict(result.representation_ratios) == {"young": 1.0, "middle": 1.0, "senior": 1.0}
    assert result.chi_squared_statistic == 0.0

    # The aggregation is arithmetically true against the detailed bands.
    detailed = DEFAULT_BENCHMARKS["age_group"]
    assert detailed["under_18"] + detailed["18-24"] == pytest.approx(0.31)
    assert detailed["25-34"] + detailed["35-44"] + detailed["45-54"] == pytest.approx(0.39)
    assert detailed["55-64"] + detailed["65+"] == pytest.approx(0.30)


def test_an_alias_spelled_dataset_finds_the_same_benchmark():
    """ "M"/"F" is one population under two spellings, and grades identically."""
    aliased = pd.DataFrame({"gender": ["M"] * 4900 + ["F"] * 5100})
    result = _detect(aliased)

    assert result.severity == RepresentationSeverity.ADEQUATE
    assert dict(result.representation_ratios) == {"female": 1.0, "male": 1.0}
    # group_distributions still carries the labels the data spells, and the
    # join between the two dicts still answers on those labels.
    assert result.group_distributions == {"F": 0.51, "M": 0.49}
    assert result.representation_ratios.get("M") == 1.0
    assert result.representation_ratios.get("F") == 1.0

    mixed_race = pd.DataFrame(
        {
            "race": ["White"] * 5760
            + ["African_American"] * 1210
            + ["Latino"] * 1870
            + ["asian"] * 590
            + ["native_american"] * 70
            + ["pacific_islander"] * 20
            + ["Mixed"] * 280
            + ["other"] * 200
        }
    )
    folded = _detect(mixed_race, "race")
    assert folded.severity == RepresentationSeverity.ADEQUATE
    assert folded.representation_ratios.get("African_American") == 1.0
    assert folded.representation_ratios.get("Latino") == 1.0


# ── over-correction controls: the negative cases still fire ────────────────


def test_a_ninety_ten_split_is_still_critical_with_the_default_table():
    """Nothing was traded away. The measured numbers, not "some severity"."""
    skewed = pd.DataFrame({"gender": ["male"] * 9000 + ["female"] * 1000})
    result = _detect(skewed)

    assert result.severity == RepresentationSeverity.CRITICAL
    assert result.representation_ratios.get("female") == pytest.approx(0.19607843137254902)
    assert result.representation_ratios.get("male") == pytest.approx(1.8367346938775508)
    assert result.chi_squared_statistic == pytest.approx(6726.6906762705075)
    assert result.chi_squared_pvalue == 0.0
    assert result.chi_squared_significant is True

    (under,) = result.underrepresented_groups
    assert under["group"] == "female"
    assert under["count"] == 1000
    assert under["deficit"] == 4100


def test_a_mild_skew_grades_medium_not_adequate_and_not_critical():
    """60/40 sits between the bands, and must land in the middle one."""
    result = _detect(pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40}))

    assert result.severity == RepresentationSeverity.MEDIUM
    assert result.representation_ratios.get("female") == pytest.approx(0.784313725490196)
    assert result.chi_squared_statistic == pytest.approx(4.841936774709884)
    assert result.chi_squared_pvalue == pytest.approx(0.027775683131414708)


def test_an_absent_benchmark_group_is_still_maximal_underrepresentation():
    """An all-male dataset against 49/51, and now WITH a statistic behind it.

    This case is why "a missing chi-squared means no verdict" would have been
    the wrong guard: the CRITICAL here is correct, and the old code reached it
    with chi_squared None because the test skipped every group the data did not
    contain. The test now spans every benchmarked category, so a group with
    zero observations contributes to it instead of silencing it.
    """
    result = _detect(pd.DataFrame({"gender": ["male"] * 100}))

    assert result.severity == RepresentationSeverity.CRITICAL
    (under,) = result.underrepresented_groups
    assert under["group"] == "female"
    assert under["ratio"] == 0.0
    assert under["absent"] is True
    assert under["deficit"] == 51

    assert result.chi_squared_statistic == pytest.approx(104.08163265306123)
    assert result.chi_squared_pvalue == pytest.approx(1.9414972989427792e-24)
    assert result.chi_squared_significant is True


def test_a_column_the_built_in_table_cannot_describe_gets_no_benchmark():
    """A gender column spelled 0/1 gets an honest refusal, not phantom groups.

    Before the fix this produced every benchmark key as an absent group. There
    is no measurement to be had here, so none is reported.
    """
    numeric = pd.DataFrame({"gender": ["0"] * 5000 + ["1"] * 5000})
    with pytest.warns(UserWarning, match="UNBENCHMARKED"):
        (result,) = detect_representation_bias(numeric, ["gender"], include_intersectional=False)

    assert result.benchmark_source is None
    assert dict(result.representation_ratios) == {}
    assert result.underrepresented_groups == []
    assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert result.chi_squared_statistic is None
    assert result.chi_squared_significant is None


# ── d: the chi-squared is three-state, and the third state says so ─────────


def test_a_chi_squared_that_could_not_run_reports_none_and_names_why():
    """Partial coverage is a could-not-check, never a measured "not significant"."""
    partial = pd.DataFrame({"gender": ["Male"] * 60 + ["Female"] * 30 + ["Non-binary"] * 10})
    with pytest.warns(UserWarning, match="UNMEASURED, not 'not significant'"):
        (result,) = detect_representation_bias(
            partial,
            ["gender"],
            benchmarks={"gender": {"male": 0.5, "female": 0.5}},
            include_intersectional=False,
        )

    assert result.chi_squared_statistic is None
    assert result.chi_squared_pvalue is None
    assert result.chi_squared_significant is None
    assert result.chi_squared_significant is not False

    # The per-group ratios ARE measurements and are still reported: the
    # unmeasurable statistic does not erase the finding it does not cover.
    assert result.representation_ratios.get("Female") == pytest.approx(0.6)
    assert result.severity == RepresentationSeverity.HIGH


def test_a_p_value_of_exactly_zero_is_significant_not_unmeasured():
    """0.0 is falsy and is the STRONGEST evidence the test can give.

    Same trap as S-01 in compare_to_benchmark, one function away: this boundary
    must read ``is not None``, never truthiness.
    """
    extreme = pd.DataFrame({"gender": ["male"] * 3000 + ["female"] * 1})
    result = _detect(extreme, benchmarks={"gender": {"male": 0.5, "female": 0.5}})

    assert result.chi_squared_pvalue == 0.0
    assert result.chi_squared_significant is True
    assert result.chi_squared_significant is not None


# ── the second finding: significance_level is honoured, or refused ─────────


def test_the_significance_verdict_turns_on_the_caller_alpha():
    """60/40 gives p = 0.0278, which straddles 0.01 and 0.05.

    The parameter was documented as "Alpha level for chi-squared test", threaded
    into _analyze_single_attribute, and never read there. Four alphas produced
    byte-identical results.
    """
    mild = pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40})

    loose = _detect(mild, significance_level=0.05)
    tight = _detect(mild, significance_level=0.01)

    assert loose.chi_squared_pvalue == pytest.approx(0.027775683131414708)
    assert tight.chi_squared_pvalue == loose.chi_squared_pvalue

    assert tight.chi_squared_pvalue > 0.01
    assert loose.chi_squared_pvalue < 0.05
    assert loose.chi_squared_significant is True
    assert tight.chi_squared_significant is False

    # The alpha that produced each verdict is reported beside it, so the two
    # cannot drift apart in a stored result.
    assert loose.significance_level == 0.05
    assert tight.significance_level == 0.01


def test_the_alpha_does_not_secretly_move_the_severity():
    """Over-correction control, and the docstring's own claim.

    ``severity`` is graded from the per-group representation ratios, not from a
    hypothesis test, and the documented scope of significance_level says so.
    A knob that quietly re-graded the verdict would be a different claims-vs-code
    defect from the inert one.
    """
    mild = pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40})
    severities = {
        alpha: _detect(mild, significance_level=alpha).severity
        for alpha in (0.001, 0.01, 0.05, 0.5, 0.99)
    }
    assert set(severities.values()) == {RepresentationSeverity.MEDIUM}


def test_an_alpha_outside_zero_to_one_is_refused():
    """A value that cannot be an alpha is refused, not silently absorbed."""
    mild = pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40})
    for bad in (0.0, 1.0, -0.1, 5, 1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            detect_representation_bias(
                mild, ["gender"], significance_level=bad, include_intersectional=False
            )

    # Control: the valid extremes are accepted and reported back unchanged.
    for good in (1e-6, 0.05, 0.999999):
        assert _detect(mild, significance_level=good).significance_level == good


# ── sibling in the same file: a crash that read as a clean result ──────────


def test_a_failed_intersectional_analysis_is_not_reported_as_clean():
    """The same swallow, one function down.

    ``_analyze_intersectional`` caught every exception, logged it at DEBUG
    (invisible by default) and returned the empty list it also returns when the
    analysis ran and found nothing. A crash and a clean intersectional result
    were byte-identical to every caller.
    """
    from vfairness.preprocessing.bias_detection.representation import _analyze_intersectional

    # value_counts on a column of lists raises TypeError: unhashable type.
    broken = pd.DataFrame({"a": [["x"], ["y"]] * 60, "b": ["p", "q"] * 60})
    with pytest.warns(UserWarning, match="UNASSESSED, not clean"):
        findings = _analyze_intersectional(
            broken, ["a", "b"], min_group_size=1, underrepresentation_threshold=0.8
        )
    assert findings == []

    # Over-correction control: a healthy frame still runs silently and still
    # finds the intersection that is genuinely underrepresented.
    healthy = pd.DataFrame(
        {
            "gender": ["male"] * 500 + ["female"] * 500,
            "race": (["white"] * 450 + ["black"] * 50) + (["white"] * 490 + ["black"] * 10),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clean = _analyze_intersectional(
            healthy, ["gender", "race"], min_group_size=5, underrepresentation_threshold=0.8
        )
    assert [f["intersection"] for f in clean] == ["female_black"]
    assert clean[0]["ratio"] == pytest.approx(0.3333333333333333)
