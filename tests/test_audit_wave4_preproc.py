"""Regression tests pinning the Wave-4 preprocessing fixes
(proxy / representation / statistical bias detection)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.proxy import (
    identify_proxy_variables,
)
from vfairness.preprocessing.bias_detection.representation import (
    RepresentationSeverity,
    detect_representation_bias,
)
from vfairness.preprocessing.bias_detection.statistical import (
    DisparityType,
    _test_categorical_disparity,
)


def _proxy_frame(n: int = 400, outcome_name: str = "outcome") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    gender = rng.choice(["male", "female"], n)
    outcome = np.where(
        gender == "male",
        rng.choice([1, 0], n, p=[0.8, 0.2]),
        rng.choice([1, 0], n, p=[0.2, 0.8]),
    )
    return pd.DataFrame(
        {
            "gender": gender,
            outcome_name: outcome,
            "noise": rng.normal(size=n),
        }
    )


# ── proxy: the outcome column must not be scanned as a proxy ────────────────


def test_outcome_column_inferred_by_name_is_excluded():
    df = _proxy_frame(outcome_name="outcome")
    with pytest.warns(UserWarning, match="outcome column"):
        results = identify_proxy_variables(df, ["gender"])
    assert "outcome" not in [r.feature for r in results]


def test_explicit_outcome_column_is_excluded():
    # 'approved' is not on the name-hint list, so exclusion must come from
    # the explicit outcome_column parameter.
    df = _proxy_frame(outcome_name="approved")
    results = identify_proxy_variables(df, ["gender"], outcome_column="approved")
    assert "approved" not in [r.feature for r in results]


def test_real_proxy_still_flagged_when_outcome_excluded():
    df = _proxy_frame(outcome_name="approved")
    rng = np.random.default_rng(1)
    # zip strongly tracks gender: a genuine proxy that must survive.
    df["zipcode"] = np.where(
        df["gender"] == "male",
        rng.choice(["10001", "10002"], len(df), p=[0.9, 0.1]),
        rng.choice(["10001", "10002"], len(df), p=[0.1, 0.9]),
    )
    results = identify_proxy_variables(df, ["gender"], outcome_column="approved")
    assert "zipcode" in [r.feature for r in results]


def test_explicit_feature_columns_respected_no_inference():
    # An explicit feature list is scanned as-is: no name-based exclusion.
    df = _proxy_frame(outcome_name="outcome")
    results = identify_proxy_variables(df, ["gender"], feature_columns=["outcome", "noise"])
    assert "outcome" in [r.feature for r in results]


# ── proxy: small samples must warn loudly, not return silently ──────────────


def test_small_attribute_sample_warns_instead_of_silent_empty():
    rng = np.random.default_rng(2)
    g = rng.choice(["a", "b"], 50)
    df = pd.DataFrame({"prot": g, "copy_of_prot": g})
    with pytest.warns(UserWarning, match="Too few samples to assess proxies"):
        results = identify_proxy_variables(df, ["prot"])
    assert results == []


def test_small_feature_overlap_warns_per_attribute():
    rng = np.random.default_rng(3)
    g = rng.choice(["a", "b"], 200)
    sparse = pd.Series(g).where(pd.Series(range(200)) < 50)  # 50 non-null
    df = pd.DataFrame({"prot": g, "sparse_copy": sparse})
    with pytest.warns(UserWarning, match="overlapping samples"):
        identify_proxy_variables(df, ["prot"])


# ── representation: min sample gate on marginal analysis ────────────────────


def test_three_rows_yield_insufficient_data_not_high():
    df = pd.DataFrame({"gender": ["male", "male", "female"]})
    with pytest.warns(UserWarning, match="UNASSESSED"):
        results = detect_representation_bias(
            df, ["gender"], benchmarks={"gender": {"male": 0.5, "female": 0.5}}
        )
    assert results[0].severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert results[0].underrepresented_groups == []
    assert any("Insufficient data" in r for r in results[0].recommendations)


def test_adequate_sample_still_gets_normal_verdict():
    df = pd.DataFrame({"gender": ["male"] * 50 + ["female"] * 50})
    results = detect_representation_bias(
        df, ["gender"], benchmarks={"gender": {"male": 0.5, "female": 0.5}}
    )
    assert results[0].severity == RepresentationSeverity.ADEQUATE


# ── representation: default benchmark labelled and warned ───────────────────


def test_default_benchmark_labelled_default_us_census_with_warning():
    """The label, the warning, AND the verdict that benchmark actually produced.

    THIS GUARD USED TO STOP AT THE LABEL. It asserted that the fallback warning
    fired and that benchmark_source read "default_us_census", and never looked
    at the severity those figures produced. It was green for the entire life of
    a defect that graded a dataset matching the built-in benchmark to the last
    row as CRITICAL: the shipped table gave every spelling of a group its own
    copy of the proportion, so gender summed to 3.000, and "m", "f", "man" and
    "woman" were reported as missing demographic groups with deficits of 1633
    and 1700 people. A guard that checks the label and not the verdict catches
    nothing, so both are checked here now, with measured numbers.
    """
    df = pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40})
    with pytest.warns(UserWarning, match="default_us_census"):
        results = detect_representation_bias(df, ["gender"], include_intersectional=False)
    r = results[0]
    assert r.benchmark_source == "default_us_census"

    # 60/40 against the built-in 49/51 is a real, MEASURED skew, and the
    # numbers say so exactly: 0.40 / 0.51 and 0.60 / 0.49.
    assert r.representation_ratios.get("female") == pytest.approx(0.784313725490196)
    assert r.representation_ratios.get("male") == pytest.approx(1.2244897959183674)
    assert r.severity == RepresentationSeverity.MEDIUM
    assert r.chi_squared_statistic == pytest.approx(4.841936774709884)
    assert r.chi_squared_pvalue == pytest.approx(0.027775683131414708)

    # No spelling variant may appear as a group of its own, in any list.
    named = {g["group"] for g in r.underrepresented_groups + r.overrepresented_groups}
    assert named == {"female", "male"}
    assert set(r.representation_ratios) == {"female", "male"}


def test_the_default_benchmark_grades_its_own_distribution_adequate():
    """4900 male + 5100 female IS the built-in 49/51 benchmark, to the last row.

    The reference case for the table defect. A clean install, correct input, a
    flagship public API, and the answer was CRITICAL with a chi-squared of None.
    """
    df = pd.DataFrame({"gender": ["male"] * 4900 + ["female"] * 5100})
    with pytest.warns(UserWarning, match="default_us_census"):
        results = detect_representation_bias(df, ["gender"], include_intersectional=False)
    r = results[0]

    assert r.benchmark_source == "default_us_census"
    assert r.severity == RepresentationSeverity.ADEQUATE
    assert dict(r.representation_ratios) == {"female": 1.0, "male": 1.0}
    assert r.underrepresented_groups == []
    assert r.overrepresented_groups == []
    # A perfect fit is a MEASURED zero, not a missing statistic.
    assert r.chi_squared_statistic == 0.0
    assert r.chi_squared_pvalue == pytest.approx(1.0)
    assert r.chi_squared_significant is False
    assert not any("CRITICAL" in rec for rec in r.recommendations)


def test_provided_benchmark_still_labelled_provided_no_default_warning():
    df = pd.DataFrame({"gender": ["male"] * 60 + ["female"] * 40})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = detect_representation_bias(
            df, ["gender"], benchmarks={"gender": {"male": 0.5, "female": 0.5}}
        )
    assert results[0].benchmark_source == "provided"
    assert not any("default_us_census" in str(w.message) for w in caught)


# ── representation: overrepresentation-only reaches OVERREPRESENTED ─────────


def test_overrepresentation_only_is_not_adequate():
    df = pd.DataFrame({"race": ["a"] * 35 + ["b"] * 35 + ["c"] * 30})
    results = detect_representation_bias(
        df, ["race"], benchmarks={"race": {"a": 0.4, "b": 0.4, "c": 0.2}}
    )
    r = results[0]
    assert r.underrepresented_groups == []
    assert [g["group"] for g in r.overrepresented_groups] == ["c"]
    assert r.severity == RepresentationSeverity.OVERREPRESENTED


def test_underrepresentation_still_outranks_overrepresentation():
    # 2 groups: one clearly under, the other over; the under-driven
    # severity must win, not OVERREPRESENTED.
    df = pd.DataFrame({"gender": ["male"] * 80 + ["female"] * 20})
    results = detect_representation_bias(
        df, ["gender"], benchmarks={"gender": {"male": 0.5, "female": 0.5}}
    )
    assert results[0].severity == RepresentationSeverity.CRITICAL


# ── statistical: CI on the resolved positive class, not the last column ─────


def test_categorical_ci_uses_positive_class_column():
    rng = np.random.default_rng(4)
    g = np.array(["groupA"] * 150 + ["groupB"] * 150)
    # 'approved' sorts BEFORE 'denied', so the last crosstab column is the
    # negative class: the old code produced a sign-flipped CI here.
    out = np.concatenate(
        [
            rng.choice(["approved", "denied"], 150, p=[0.8, 0.2]),
            rng.choice(["approved", "denied"], 150, p=[0.3, 0.7]),
        ]
    )
    df = pd.DataFrame({"prot": g, "decision": out})
    result = _test_categorical_disparity(df, "decision", "prot", DisparityType.OUTCOME, {}, 30)
    assert result is not None
    p_a = result.group_statistics["groupA"]["positive_rate"]
    p_b = result.group_statistics["groupB"]["positive_rate"]
    diff = p_a - p_b
    lo, hi = result.confidence_interval
    assert lo is not None and hi is not None
    # The CI must straddle the point estimate computed on the same
    # positive-class column (row order in the crosstab is groupA, groupB).
    assert lo <= diff <= hi
    assert (lo + hi) / 2 == pytest.approx(diff, abs=1e-9)
