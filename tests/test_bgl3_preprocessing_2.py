"""BGL-3 preprocessing-2: does each unit refuse honestly when nothing is measurable?

Batch preprocessing-2 covers 14 public units across three files:
``preprocessing/bias_detection/geographic_data.py``,
``preprocessing/bias_detection/statistical.py`` and
``preprocessing/feature_engineering/data_balancing.py``.

Every assertion here was produced by running the real function on an input where
the quantity it reports genuinely does not exist, and every docstring records
what the unit answered BEFORE the fix, in the numbers it actually printed. The
control tests are load-bearing in the other direction: a unit that refuses
everything is as wrong as one that answers everything, and the literals below
were captured from the healthy runs so a fix that refuses too much fails here.
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.geographic_data import (
    HOLCGrade,
    assess_geographic_feature_risk,
    get_available_holc_cities,
    get_holc_risk_level,
    get_svi_data_url,
    parse_holc_grade,
)
from vfairness.preprocessing.bias_detection.statistical import (
    _interpret_eta_squared,
    compute_effect_sizes,
    detect_specification_bias,
    detect_temporal_drift,
    run_disparity_tests,
)
from vfairness.preprocessing.feature_engineering.data_balancing import (
    SMOTEResampler,
    propensity_weights,
)


def _caught(fn):
    """Run ``fn`` and return (result, list of warning messages)."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in seen]


def _joined(messages: List[str]) -> str:
    return " || ".join(messages)


# ===========================================================================
# 1. compute_effect_sizes: a group with nothing measured is not a group
# ===========================================================================


def _two_groups_one_unmeasured() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "score": list(np.linspace(0.0, 1.0, 40)) + [np.nan] * 40,
        }
    )


def test_a_group_with_no_measured_value_is_reported_not_silently_compared():
    """An all-NaN group used to pass the "2 groups with data" guard and be graded.

    Measured 2026-09-27 on this exact frame (40 rows of 'a' over 0.0 to 1.0
    beside 40 rows of 'b' whose score was entirely unmeasured), before the fix:

        {'effect_sizes': {'cohens_d': {'value': nan,
                                       'interpretation': 'not_measurable'},
                          'eta_squared': {'value': nan,
                                          'interpretation': 'large'}}}

    with no warning at all. The guard counted dict KEYS, so an empty array
    satisfied "fewer than 2 groups with data", and one result object then
    carried both could-not-check and the strongest magnitude band on the scale
    for the same unmeasurable comparison.
    """
    result, messages = _caught(
        lambda: compute_effect_sizes(_two_groups_one_unmeasured(), "score", "g")
    )

    assert result["groups_without_data"] == ["b"]
    assert result["error"] == "Fewer than 2 groups with data"
    assert result["effect_sizes"] == {}, (
        "an effect size was published for a comparison with one group of data"
    )
    assert "have no measured value of 'score'" in _joined(messages)
    assert "could-not-check" in _joined(messages)


def test_the_groups_that_do_have_data_are_still_measured():
    """Not over-refused: with a, b measured and c empty, a, b are still compared.

    Dropping the empty group must not turn into refusing the whole comparison.
    """
    df = pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            "score": ([0.80, 0.85, 0.90, 0.95] * 10)
            + ([0.10, 0.15, 0.20, 0.25] * 10)
            + [np.nan] * 40,
        }
    )

    result, messages = _caught(lambda: compute_effect_sizes(df, "score", "g"))

    assert result["groups_without_data"] == ["c"]
    assert "error" not in result
    d = result["effect_sizes"]["cohens_d"]
    assert np.isfinite(d["value"]) and abs(d["value"]) > 0.8
    assert d["interpretation"] == "large"
    assert "['c']" in _joined(messages)


def test_control_a_healthy_effect_size_run_measures_and_says_nothing():
    """Over-correction control. A warning on every run is a warning on none.

    The literals come from the same fixture before the change, so a fix that
    perturbed the measured branch fails here.
    """
    continuous = pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "score": ([0.80, 0.85, 0.90, 0.95] * 10) + ([0.10, 0.15, 0.20, 0.25] * 10),
        }
    )
    result, messages = _caught(lambda: compute_effect_sizes(continuous, "score", "g"))
    assert messages == [], f"the healthy run warned: {messages}"
    assert "groups_without_data" not in result
    assert result["effect_sizes"]["cohens_d"]["value"] == pytest.approx(
        12.364465212858986, rel=1e-9, abs=0.0
    )
    assert result["effect_sizes"]["cohens_d"]["interpretation"] == "large"
    assert result["effect_sizes"]["eta_squared"]["interpretation"] == "large"

    categorical = pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 100,
            "approved": ["yes"] * 85 + ["no"] * 15 + ["yes"] * 15 + ["no"] * 85,
        }
    )
    result2, messages2 = _caught(lambda: compute_effect_sizes(categorical, "approved", "g"))
    assert messages2 == [], f"the healthy categorical run warned: {messages2}"
    assert result2["effect_sizes"]["cramers_v"]["value"] == pytest.approx(0.69, rel=1e-9, abs=0.0)
    assert result2["effect_sizes"]["cramers_v"]["interpretation"] == "large"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "not_measurable"),
        (float("inf"), "not_measurable"),
        (None, "not_measurable"),
        (0.005, "negligible"),
        (0.03, "small"),
        (0.10, "medium"),
        (0.50, "large"),
    ],
)
def test_an_unmeasurable_eta_squared_is_not_graded_large(value, expected):
    """A NaN eta-squared was published as 'large', the top band on the scale.

    Every band in this helper is a ``<`` comparison and every comparison against
    NaN is False, so a NaN fell through to the final ``else``. Measured
    2026-09-27: ``_interpret_eta_squared(float('nan'))`` returned ``'large'``,
    while the sibling ``_interpret_effect_size`` on the same statistic returned
    ``'not_measurable'`` because its guard sits above its dispatch. The four
    finite rows here keep the real bands pinned so the guard cannot swallow
    them.
    """
    assert _interpret_eta_squared(value) == expected


# ===========================================================================
# 2. run_disparity_tests: a group dropped for size must be disclosed
# ===========================================================================


def _three_groups_smallest_all_rejected() -> pd.DataFrame:
    """60 + 60 rows at a 50% approval rate, plus 5 rows of 'c' all rejected."""
    return pd.DataFrame(
        {
            "g": ["a"] * 60 + ["b"] * 60 + ["c"] * 5,
            "approved": [1] * 30 + [0] * 30 + [1] * 30 + [0] * 30 + [0] * 5,
        }
    )


def test_a_group_below_min_group_size_is_disclosed_not_dropped_silently():
    """The only group with a disparity was dropped and the rest reported clean.

    Measured 2026-09-27 on this frame, before the fix::

        test_name='Chi-squared test'  pvalue=1.0  effect_size=0.0
        effect_interpretation='negligible'  disparity_magnitude=0.0
        sample_sizes={'a': 60, 'b': 60}   warnings: []

    p=1.0 and effect 0.0 are the strongest all-clear the scale can give, and
    group 'c' (5 of 5 rejected) appeared nowhere in the payload. This entry
    point was the one caller of ``_test_disparity`` that passed no ``excluded``
    list, so the exclusions the helper collects were thrown away;
    ``analyze_statistical_disparities`` has warned about the identical list
    since audit 6.
    """
    result, messages = _caught(
        lambda: run_disparity_tests(_three_groups_smallest_all_rejected(), "approved", "g")
    )

    # The measured half is unchanged: the two testable groups really are alike.
    assert result["test_name"] == "Chi-squared test"
    assert result["pvalue"] == pytest.approx(1.0, rel=1e-9, abs=0.0)
    assert result["sample_sizes"] == {"a": 60, "b": 60}

    # The could-not-check half is now readable.
    assert result["min_group_size"] == 10
    assert len(result["groups_excluded"]) == 1
    assert "c (n=5)" in result["groups_excluded"][0]
    warned = _joined(messages)
    assert "excluded 1 group comparison(s)" in warned
    assert "NOT TESTED" in warned
    assert "NOT evidence that they are free of disparity" in warned


def test_the_could_not_check_dict_names_the_groups_it_excluded():
    """No test ran at all. The third state keeps its shape and gains the reason.

    Before the fix this returned exactly four keys and no warning, so "both
    groups were too small" and "the column does not exist" were the same
    output.
    """
    tiny = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    result, messages = _caught(lambda: run_disparity_tests(tiny, "y", "g"))

    assert result["error"] == "Could not run disparity test"
    assert result["test_name"] is None
    assert "pvalue" not in result
    assert "a (n=5), b (n=5)" in result["groups_excluded"][0]
    assert "fewer than 2 groups remained" in result["groups_excluded"][0]
    assert "excluded 1 group comparison(s)" in _joined(messages)


def test_control_a_two_group_run_excludes_nothing_and_is_silent():
    """Over-correction control, with the measured numbers pinned."""
    df = pd.DataFrame(
        {
            "g": ["a"] * 60 + ["b"] * 60,
            "approved": [1] * 51 + [0] * 9 + [1] * 9 + [0] * 51,
        }
    )
    result, messages = _caught(lambda: run_disparity_tests(df, "approved", "g"))

    assert messages == [], f"the healthy run warned: {messages}"
    assert result["groups_excluded"] == []
    assert result["test_name"] == "Chi-squared test"
    assert result["significance"] == "highly_significant"
    assert result["effect_size"] == pytest.approx(0.6833333333333333, rel=1e-9, abs=0.0)


# ===========================================================================
# 3. detect_temporal_drift: "pass" with nothing assessed, and a degenerate PSI
# ===========================================================================


def _dated(race: List[Any]) -> pd.DataFrame:
    n = len(race)
    return pd.DataFrame(
        {
            "application_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "race": race,
            "approved": [1, 0] * (n // 2),
        }
    )


@pytest.mark.parametrize(
    ("attributes", "frame", "expected_in_reason"),
    [
        (["gender"], _dated(["a", "b"] * 100), "not column(s) of the frame"),
        ([], _dated(["a", "b"] * 100), "no protected attribute was named"),
        (
            ["race"],
            _dated(["g%03d" % i for i in range(200)]),
            "identifier-grade for PSI",
        ),
    ],
)
def test_no_attribute_assessed_is_unavailable_not_pass(attributes, frame, expected_in_reason):
    """Zero attributes assessed used to be published as a clean drift screen.

    Measured 2026-09-27, all three routes into the same return::

        {'available': True, 'timeColumn': 'application_date', 'drift': [],
         'skippedHighCardinality': [...], 'severity': 'pass',
         'summary': 'No material temporal drift in the assessed attributes
                     over application_date (static-snapshot screen only).'}

    The assessed attributes were none. ``available: False`` with a reason is the
    state this function already uses for a missing time column, so it is the one
    this belongs in too.
    """
    result, messages = _caught(
        lambda: detect_temporal_drift(frame, attributes, prediction="approved")
    )

    assert result["available"] is False
    assert result["drift"] == []
    assert expected_in_reason in result["reason"]
    assert "nothing was compared, so nothing is clean" in result["reason"]
    assert "no protected attribute could be assessed" in _joined(messages)


def test_a_degenerate_psi_is_not_assessed_rather_than_a_pass():
    """A population replaced wholesale was graded 'pass' by the guard for it.

    Measured 2026-09-27 on 200 rows whose 20-level attribute shares NO value
    between the early and late cohort::

        {'attribute': 'race', 'compositionPSI': 23.0256, 'severity': 'pass',
         'plain': '... the race mix shifted materially (PSI 23.03) ...'}
        top-level severity 'pass'
        summary 'No material temporal drift in the assessed attributes ...'

    One entry claimed a material shift and a pass at the same time. The >= 3.0
    branch exists so a degenerate PSI cannot falsely escalate to 'critical'
    (the pulse orchestrator raises a finding on warn / critical only), and that
    is preserved: 'not_assessed' escalates nothing and claims nothing.
    """
    early = ["c%02d" % (i % 10) for i in range(100)]
    late = ["c%02d" % (10 + i % 10) for i in range(100)]

    result, messages = _caught(
        lambda: detect_temporal_drift(_dated(early + late), ["race"], prediction="approved")
    )

    entry = result["drift"][0]
    assert entry["compositionPSI"] == pytest.approx(23.0256, abs=1e-4)
    assert entry["severity"] == "not_assessed"
    assert "share only 0 of 20 observed race value(s)" in entry["notAssessedReason"]
    assert "NOT assessed" in entry["plain"]
    assert result["severity"] == "not_assessed"
    assert "No material temporal drift" not in result["summary"]
    assert "could-not-check" in result["summary"]
    assert "NOT graded" in _joined(messages)


def test_control_drift_is_still_critical_and_stability_is_still_pass():
    """Over-correction control for both directions of the drift screen."""
    drifting = _dated(["a"] * 80 + ["b"] * 20 + ["a"] * 20 + ["b"] * 80)
    result, messages = _caught(
        lambda: detect_temporal_drift(drifting, ["race"], prediction="approved")
    )
    assert messages == [], f"the drifting run warned: {messages}"
    assert result["available"] is True
    assert result["severity"] == "critical"
    assert result["drift"][0]["compositionPSI"] == pytest.approx(1.6636, abs=1e-4)
    assert "Temporal drift risk" in result["summary"]

    stable = _dated(["a", "b"] * 100)
    result2, messages2 = _caught(
        lambda: detect_temporal_drift(stable, ["race"], prediction="approved")
    )
    assert messages2 == [], f"the stable run warned: {messages2}"
    assert result2["available"] is True
    assert result2["severity"] == "pass"
    assert "No material temporal drift" in result2["summary"]


# ===========================================================================
# 4. detect_specification_bias: three checks, and none of them ran
# ===========================================================================


_SPEC_FRAME = pd.DataFrame(
    {"x": np.linspace(0, 1, 50), "cost": np.linspace(0, 1, 50), "approved": [1, 0] * 25}
)


@pytest.mark.parametrize(
    ("kwargs", "frame"),
    [
        ({}, _SPEC_FRAME),
        ({"outcome": "nope"}, _SPEC_FRAME),
        ({"outcome": "y"}, pd.DataFrame()),
    ],
)
def test_a_screen_that_examined_nothing_is_unavailable_not_pass(kwargs, frame):
    """All three of these examined nothing and every one returned 'pass'.

    Measured 2026-09-27, identical output for each::

        {'available': True, 'findings': [], 'severity': 'pass',
         'summary': 'No specification / construct-validity red flag detected
                     on this data.'}

    The screen runs three checks and each needs an input it may not have been
    given: a column NAME for the proxy-target check, a numeric outcome for the
    leakage correlation, and both an outcome and a prediction for the
    circular-evaluation check. With no check able to run, an empty findings list
    means "nothing was examined", which is not a clean bill of health.
    """
    result, messages = _caught(lambda: detect_specification_bias(frame, **kwargs))

    assert result["available"] is False
    assert result["findings"] == []
    assert result["checksRun"] == []
    assert len(result["checksNotRun"]) == 3
    assert "nothing was examined and nothing is clean" in result["reason"]
    assert "no specification / construct-validity check could run" in _joined(messages)
    assert "severity" not in result, "an unavailable screen must not publish a verdict"


def test_a_non_numeric_outcome_discloses_the_leakage_scan_it_could_not_run():
    """The leakage scan is a Pearson correlation and a string outcome disables it.

    Measured 2026-09-27 on outcome='decision' holding 'yes'/'no' beside a
    feature column that encodes the same decision as 1/0: the screen returned
    ``{'available': True, 'findings': [], 'severity': 'pass', 'summary': 'No
    specification / construct-validity red flag detected on this data.'}``
    having correlated nothing at all. Encoding the same frame numerically finds
    the leak at |r|=1.00, so the 'pass' was a statement about a scan that never
    happened.
    """
    text = pd.DataFrame({"decision": ["yes", "no"] * 25, "leak_num": [1, 0] * 25})

    result, messages = _caught(lambda: detect_specification_bias(text, outcome="decision"))

    assert result["available"] is True
    assert result["findings"] == []
    assert result["checksRun"] == ["proxy_target"]
    not_run = " ".join(result["checksNotRun"])
    assert "target_leakage" in not_run and "reads as a number" in not_run
    assert "check(s) that ran" in result["summary"]
    assert "could not run" in _joined(messages)

    # The same data, numerically encoded, IS scanned: the refusal above is
    # about the encoding and not about the screen having given up.
    numeric = pd.DataFrame({"decision": [1, 0] * 25, "leak_num": [1, 0] * 25})
    found, _ = _caught(lambda: detect_specification_bias(numeric, outcome="decision"))
    assert found["severity"] == "critical"
    assert found["findings"][0]["kind"] == "target_leakage"


def test_control_the_screen_still_finds_a_proxy_target_and_a_leak():
    """Over-correction control: the findings path is untouched and silent."""
    result, messages = _caught(
        lambda: detect_specification_bias(_SPEC_FRAME, outcome="cost", prediction="approved")
    )

    assert messages == [], f"the run that found something warned: {messages}"
    assert result["available"] is True
    assert result["severity"] == "critical"
    kinds = sorted(f["kind"] for f in result["findings"])
    assert kinds == ["proxy_target", "target_leakage"]
    assert result["checksRun"] == [
        "proxy_target",
        "target_leakage (1 feature column(s) scanned)",
        "circular_evaluation",
    ]
    assert result["checksNotRun"] == []


# ===========================================================================
# 5. propensity_weights: the neutral 1.0 for a row with no recorded group
# ===========================================================================


def _single_group_with_unrecorded_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a"] * 15 + [None] * 5,
            "x": np.linspace(-1.0, 1.0, 20),
        }
    )


def test_single_group_gives_the_identity_weight_only_to_a_recorded_group():
    """The warning said NaN, the docstring said NaN, and the array said 1.0.

    Measured 2026-09-27 on 15 rows of group 'a' beside 5 rows whose protected
    value was never recorded::

        sample_weights = [1.0] * 20
        warning: "5 of 20 row(s) have no recorded value for the protected
                  attribute(s); ... their sample weight is NaN"

    ``np.ones(n)`` covered every row, so the five rows nobody could place in a
    group received the neutral training weight and would have trained a model as
    though they had been measured. This is the exact substitution
    ``_weights_from_scores`` refuses one function away.
    """
    result, messages = _caught(
        lambda: propensity_weights(_single_group_with_unrecorded_rows(), ["g"])
    )

    weights = np.asarray(result["sample_weights"], dtype=float)
    assert result["propensity_model"] == "single_group"
    assert list(weights[:15]) == [1.0] * 15
    assert np.isnan(weights[15:]).all(), f"unrecorded rows kept a weight: {weights[15:]}"
    assert result["n_rows_without_a_recorded_group"] == 5
    assert result["group_balance"]["max_dev_after"] is None
    assert "their sample weight is NaN" in _joined(messages)


def test_control_a_pure_single_group_frame_keeps_the_identity_weight():
    """Not over-refused: with one group and no missing value, 1.0 is correct.

    With one observed group there is nothing to rebalance, so the identity
    weight is the complete answer for every row. Only the propensity and the
    balance score are could-not-checks.
    """
    df = pd.DataFrame({"g": ["a"] * 20, "x": np.linspace(-1.0, 1.0, 20)})

    result, _ = _caught(lambda: propensity_weights(df, ["g"]))

    assert result["propensity_model"] == "single_group"
    assert list(np.asarray(result["sample_weights"], dtype=float)) == [1.0] * 20
    assert np.isnan(np.asarray(result["propensity_scores"], dtype=float)).all()
    assert result["group_balance"]["max_dev_after"] is None


def test_control_a_two_group_frame_is_still_fitted_and_measured():
    """Over-correction control for the whole propensity family."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "g": ["a"] * 30 + ["b"] * 10,
            "x": list(rng.normal(0, 1, 30)) + list(rng.normal(1.5, 1, 10)),
        }
    )

    result, messages = _caught(lambda: propensity_weights(df, ["g"]))

    assert messages == [], f"the healthy run warned: {messages}"
    assert result["propensity_model"] == "fitted"
    weights = np.asarray(result["sample_weights"], dtype=float)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert result["group_balance"]["max_dev_before"] == pytest.approx(0.25, abs=1e-9)
    assert result["group_balance"]["max_dev_after"] == pytest.approx(0.0087363822, abs=1e-8)


# ===========================================================================
# 6. SyntheticResampler.get_resampled_data: a no-op that reported success
# ===========================================================================


def _resampled(df: pd.DataFrame, y: np.ndarray):
    r = SMOTEResampler(protected_attributes=["g"], random_state=42).fit(df, y)
    (X_res, y_res), messages = _caught(lambda: r.get_resampled_data(df, y))
    return r, X_res, y_res, messages


def test_a_resample_that_added_no_row_says_so():
    """10 rows in, 10 rows out, ``fit_result.warnings == []``.

    Measured 2026-09-27 on 10 rows of one group carrying one label: the frame
    came straight back with no disclosure of any kind, byte-indistinguishable
    from a successful balance. The sibling ``CounterfactualAugmenter`` discloses
    the same degeneracy on the same shape of frame through ``_warn_no_twins``,
    and every other refusal in this class already reaches
    ``fit_result.warnings``; these two returns were the ones that did not.
    """
    df = pd.DataFrame({"g": ["a"] * 10, "f0": np.linspace(0.0, 1.0, 10)})
    r, X_res, _, _ = _resampled(df, np.ones(10, dtype=int))

    assert len(X_res) == len(df)
    warned = _joined(r.fit_result.warnings)
    assert "no synthesis was performed" in warned
    assert "single (group, label) cell" in warned
    assert "returned UNCHANGED" in warned


def test_an_empty_frame_is_disclosed_rather_than_returned_silently():
    """0 rows in, 0 rows out, ``fit_result.warnings == []`` before the fix."""
    df = pd.DataFrame({"g": pd.Series([], dtype=object), "f0": pd.Series([], dtype=float)})
    r, X_res, _, _ = _resampled(df, np.asarray([], dtype=int))

    assert len(X_res) == 0
    assert "the frame holds no rows" in _joined(r.fit_result.warnings)


def test_control_a_real_resample_balances_and_stays_silent():
    """Over-correction control: a run that DID synthesise must not warn."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g": ["a"] * 30 + ["b"] * 6,
            "f0": rng.normal(size=36),
            "f1": rng.normal(size=36),
        }
    )
    y = np.asarray([0] * 15 + [1] * 15 + [0] * 3 + [1] * 3)

    r, X_res, y_res, messages = _resampled(df, y)

    assert len(X_res) == 60 and len(y_res) == 60
    assert X_res["g"].value_counts().to_dict() == {"a": 30, "b": 30}
    assert r.fit_result.warnings == [], f"the healthy run warned: {r.fit_result.warnings}"
    assert messages == []


# ===========================================================================
# 7. geographic_data: correct refusals, pinned so they cannot regress
# ===========================================================================


@pytest.mark.parametrize(
    ("values", "kwargs", "expected_in_warning"),
    [
        ([], {}, "no geographic values were given"),
        (["90210"] * 50, {}, "none of the 50 value(s) resolved to a HOLC grade"),
        (["48201"], {"feature_type": "census_tract"}, "is not supported"),
    ],
)
def test_geographic_feature_risk_answers_nan_and_unknown_not_zero(
    values, kwargs, expected_in_warning
):
    """Verified CORRECT by execution: three could-not-checks, all NaN and loud.

    A risk_score of 0.0 is the "no risk" end of this scale, and audit 6 lane 2
    replaced it with NaN plus the verdict 'unknown' on every one of these
    paths. Pinned here because nothing else in the suite holds it: an
    unresolvable ZIP is a could-not-check, never grade A.
    """
    assessment, messages = _caught(lambda: assess_geographic_feature_risk(values, **kwargs))

    assert np.isnan(assessment.risk_score)
    assert assessment.disparate_impact_risk == "unknown"
    assert expected_in_warning in _joined(messages)


def test_control_geographic_feature_risk_still_grades_a_redlined_feature():
    """Over-correction control: 100 grade-D ZIP codes score 1.0 and 'critical'."""
    assessment, messages = _caught(lambda: assess_geographic_feature_risk(["48201"] * 100))

    assert messages == [], f"the healthy run warned: {messages}"
    assert assessment.risk_score == pytest.approx(1.0, abs=1e-12)
    assert assessment.disparate_impact_risk == "critical"
    assert assessment.holc_coverage == pytest.approx(1.0, abs=1e-12)
    assert assessment.affected_samples == 100


def test_the_holc_lookup_helpers_keep_an_explicit_unknown_state():
    """Verified CORRECT by execution: the enum and the risk map both refuse.

    ``parse_holc_grade`` answers UNKNOWN for anything outside A to D rather
    than guessing a grade, and ``get_holc_risk_level`` answers 'unknown' for
    it rather than 'low'. 'low' would be the clean end of the risk scale for a
    neighbourhood nobody classified.
    """
    assert parse_holc_grade("d") is HOLCGrade.D
    assert parse_holc_grade("Z") is HOLCGrade.UNKNOWN
    assert parse_holc_grade("") is HOLCGrade.UNKNOWN
    assert get_holc_risk_level(HOLCGrade.UNKNOWN) == "unknown"
    assert get_holc_risk_level(HOLCGrade.D) == "critical"
    assert get_holc_risk_level(HOLCGrade.A) == "low"
    # A value that is not a grade at all is 'unknown', not the clean end.
    assert get_holc_risk_level("D") == "unknown"


def test_the_city_list_and_svi_url_are_the_fixed_tables_they_claim_to_be():
    """Verified CORRECT by execution: neither unit measures anything.

    ``get_available_holc_cities`` reads one module-level table and
    ``get_svi_data_url`` formats one string, so neither has a quantity that
    could go unmeasurable. What IS worth holding is that the list is not
    empty (an empty list would read as "no city has HOLC data") and that it
    agrees with the table the ZIP lookup uses.
    """
    cities = get_available_holc_cities()
    assert len(cities) >= 39
    assert cities == sorted(cities)
    assert "Detroit_MI" in cities and "Chicago_IL" in cities
    assert get_svi_data_url(2022).endswith("data_2022_download.html")


def _unused(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
    """Keep the Dict/Any imports honest for the type checkers in CI."""
    return {}
