"""BGL-final d02: three states at the public entry for two capabilities.

Group d02 covers ``compute_feature_correlations``
(``preprocessing.feature_engineering.correlation``) and ``power_analysis``
(``operations.experimentation.power``). Both had already been through repair
passes when this file was written, so most of what is pinned here was found
ALREADY CORRECT and is pinned because an unpinned correct behaviour is one
refactor from regressing. Two defects still reproduced at a public entry and
are pinned by their real before/after:

  * ``bias_detection.proxy._correlation_ratio_with_pvalue`` returned the
    arithmetic 1.0 for a one-row-per-level partition, and
    ``identify_proxy_variables`` published it as a CRITICAL proxy. The guard had
    been applied to one of the two copies of this function.
  * ``FairnessExperiment.calculate_intersectional_power`` counted ROWS, so an
    intersection whose outcome column was entirely missing was graded
    identically to a fully observed one, and ``run_full_analysis()`` wrote that
    number into ``ExperimentResult.power_results``.

Every refusal test here is paired with a control on healthy data that STILL
MEASURES, because a guard that refuses everything passes a refusal test and
destroys the capability.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vfairness.operations.experimentation.experiment import FairnessExperiment
from vfairness.operations.experimentation.power import (
    FairnessPowerAnalyzer,
    PowerConfig,
    SPRTDecision,
)
from vfairness.preprocessing.bias_detection import proxy as bd_proxy
from vfairness.preprocessing.feature_engineering import compute_feature_correlations
from vfairness.preprocessing.feature_engineering import correlation as fe_corr

# Both copies of the correlation-ratio helper. Anything asserted about one is
# asserted about the other: the whole point of the d02 finding is that a guard
# reached one copy and not the twin the proxy detector actually calls.
_ETA_MODULES = [fe_corr, bd_proxy]
_ETA_IDS = ["feature_engineering.correlation", "bias_detection.proxy"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _one_row_per_level_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """One distinct 'race' label per row: eta is 1.0 by arithmetic, not data."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race": [f"r{i}" for i in range(n)],
            "age": rng.normal(45, 12, n),
        }
    )


def _genuine_proxy_frame(n: int = 150, seed: int = 3) -> pd.DataFrame:
    """A real continuous proxy for a 3-level categorical attribute."""
    rng = np.random.default_rng(seed)
    race = np.resize(np.array(["Black", "White", "Asian"]), n)
    offsets = {"Black": 0.0, "White": 30.0, "Asian": 60.0}
    return pd.DataFrame(
        {
            "race": race,
            "income": np.array([offsets[r] for r in race]) + rng.normal(0, 1.0, n),
        }
    )


def _s2g07_frame(seed: int, n: int = 60) -> pd.DataFrame:
    """The fixture whose control numbers item 6 of the audit corrects."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "customer_id": [f"id{i}" for i in range(n)],
            "city": np.resize(np.array(["A", "B", "C"]), n),
        }
    )


def _arm(n_per_group: int, seed: int, outcome: str) -> pd.DataFrame:
    """One experiment arm, 'm' and 'f', with a chosen amount of outcome data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"gender": np.repeat(["m", "f"], n_per_group)})
    if outcome == "real":
        df["y"] = rng.normal(0, 1, len(df))
    elif outcome == "none":
        df["y"] = np.nan
    elif outcome == "five":
        col = np.full(len(df), np.nan)
        col[:5] = rng.normal(0, 1, 5)
        df["y"] = col
    else:  # pragma: no cover - guard against a typo in a future fixture
        raise ValueError(outcome)
    return df


# ---------------------------------------------------------------------------
# 1. The twin that still returned the fabricated 1.0 (DEFECT, item 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
def test_one_row_per_level_is_refused_in_both_copies_of_the_helper(mod):
    """BEFORE (bias_detection.proxy, measured 2026-09-17):
    ``_correlation_ratio_with_pvalue(arange(1,16), [g0..g14]) == (1.0, nan)``.
    ``feature_engineering.correlation`` already returned ``(nan, nan)``.

    eta is the share of variance lying BETWEEN levels. With one observation per
    level ``ss_between`` equals ``ss_total`` identically, so 1.0 comes out for
    ANY fifteen numbers; it is a property of the partition, not of the data.
    """
    values = pd.Series(np.arange(1.0, 16.0))
    groups = pd.Series([f"g{i}" for i in range(15)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eta, pval = mod._correlation_ratio_with_pvalue(values, groups)

    assert math.isnan(eta), f"eta={eta} is arithmetic, not a measurement, on this partition"
    assert math.isnan(pval), f"a p-value of {pval} was asserted for a test that did not run"
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught), (
        "the refusal must name itself to the caller, not only return nan"
    )


def test_a_unique_per_row_attribute_is_no_longer_published_as_a_critical_proxy():
    """PUBLIC ENTRY, ``bias_detection.proxy.identify_proxy_variables``.

    BEFORE (measured 2026-09-17, 120 rows, one 'race' label per row, age from
    ``default_rng(0).normal(45, 12, 120)``): one result, correlation=1.0,
    risk_level='critical', pvalue=None, recommendation "CRITICAL: 'age' is a
    strong proxy for 'race' (correlation: 1.00). Removing this feature is
    strongly recommended."

    AFTER: zero results, with a warning saying the pair was NOT SCREENED. The
    pair is a could-not-check, and a could-not-check is not a CRITICAL finding.
    """
    df = _one_row_per_level_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = bd_proxy.identify_proxy_variables(df, ["race"], min_sample_size=100)

    assert [r.correlation for r in results] == [], (
        f"a proxy finding was published from a one-row-per-level partition: "
        f"{[(r.feature, r.correlation, r.risk_level.value) for r in results]}"
    )
    assert any("NOT SCREENED" in str(w.message) for w in caught), (
        "the pair must be disclosed as unscreened, never silently dropped"
    )


def test_control_a_genuine_continuous_proxy_still_reads_critical_in_proxy_py():
    """OVER-CORRECTION CONTROL for the guard just added to bias_detection.proxy.

    Measured after the change: correlation 0.9991, risk 'critical',
    p = 7.0e-201. If the guard ever starts refusing real partitions, the
    textbook proxy case (income proxying for race) disappears from the screen
    and this test goes red.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = bd_proxy.identify_proxy_variables(
            _genuine_proxy_frame(), ["race"], min_sample_size=100
        )

    assert len(results) == 1, f"the genuine proxy was lost: {results}"
    found = results[0]
    assert found.feature == "income"
    assert found.correlation > 0.99
    assert found.risk_level is bd_proxy.ProxyRiskLevel.CRITICAL
    assert found.pvalue is not None and found.pvalue < 1e-50


# ---------------------------------------------------------------------------
# 2. The two refusal branches, pinned SEPARATELY (item 4)
# ---------------------------------------------------------------------------
#
# Branch (a) "no level holds two observations" and branch (b) "levels outnumber
# the pairs" share one `if`, and (a) can never fire without (b) also being true
# (all-singleton means len(groups) == len(vals) > len(vals)/2). Deleting (a)
# therefore changes no return value anywhere, which is why the suite stayed
# green without it. What (a) does change is the REASON given to the reader, so
# that is what is pinned: each fixture must get its own rationale. Delete (a)
# and the singleton fixture gets the (b) wording; delete (b) and the 31-level
# fixture returns a measured eta. Each half now fails on its own.


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
def test_branch_a_only_every_observation_is_its_own_level(mod):
    values = pd.Series(np.linspace(1.0, 2.0, 40))
    groups = pd.Series([f"g{i}" for i in range(40)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eta, _ = mod._correlation_ratio_with_pvalue(values, groups)
    messages = " | ".join(str(w.message) for w in caught)

    assert math.isnan(eta)
    assert "no level holds more than one observation" in messages, (
        f"branch (a)'s rationale is missing; a reader was told instead: {messages}"
    )


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
def test_branch_b_only_thirty_one_levels_over_sixty_observations(mod):
    """Largest level holds 2, so branch (a) does NOT fire; 31 > 60/2 does."""
    levels = [f"L{i}" for i in range(29) for _ in range(2)] + ["S1", "S2"]
    values = pd.Series(np.random.default_rng(5).normal(0, 1, 60))
    assert len(set(levels)) == 31 and len(levels) == 60
    assert max(levels.count(x) for x in set(levels)) == 2, "branch (a) must not be reachable here"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eta, _ = mod._correlation_ratio_with_pvalue(values, pd.Series(levels))
    messages = " | ".join(str(w.message) for w in caught)

    assert math.isnan(eta), f"eta={eta} was reported from 31 levels over 60 observations"
    assert "31 levels over 60 observations" in messages, (
        f"branch (b)'s rationale is missing; a reader was told instead: {messages}"
    )


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
def test_control_thirty_levels_over_sixty_observations_is_still_measured(mod):
    """THE OTHER SIDE OF BRANCH (b). 30 levels over 60 rows is NOT > 60/2, so
    it must still be measured. One level fewer than the refusal fixture above:
    a guard that swallows this one has been widened past its justification."""
    levels = [f"L{i}" for i in range(30) for _ in range(2)]
    values = pd.Series(np.random.default_rng(5).normal(0, 1, 60))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta, pval = mod._correlation_ratio_with_pvalue(values, pd.Series(levels))

    assert math.isfinite(eta) and 0.0 < eta < 1.0, f"a measurable partition was refused: {eta}"
    assert math.isfinite(pval)


# ---------------------------------------------------------------------------
# 3. The ANOVA refusal branch (item 1): a fixture that REACHES f_oneway
# ---------------------------------------------------------------------------
#
# The named regression test's fixture (fifteen singleton groups) no longer
# reaches `stats.f_oneway` in feature_engineering.correlation: the partition
# guard above it returns first, so its NaN-p-value assertion passes without
# exercising the branch it is named for. Post-guard, f_oneway's own two NaN
# cases are both unreachable (every array length 1, and a wholly constant
# sample), so the only honest way to reach the branch with a VALID partition is
# to make f_oneway itself fail. Both failure modes are covered. The subject is
# unchanged: a p-value that could not be computed must be NaN, never 1.0.


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
@pytest.mark.parametrize("failure", ["nan_pvalue", "raises"])
def test_an_anova_that_did_not_run_gives_nan_with_a_measurable_partition(mod, failure, monkeypatch):
    """The partition here is perfectly measurable (2 levels, 20 observations),
    so execution reaches ``stats.f_oneway`` and eta is a real number. Only the
    ANOVA fails. A forced 1.0 there is a claim of "definitely not significant"
    about a test that never ran, and that is what silently deleted a CRITICAL
    race proxy at the caller's significance gate on 2026-09-10.
    """
    if failure == "nan_pvalue":

        def _fake(*groups):
            return stats._stats_py.F_onewayResult(float("nan"), float("nan"))

    else:

        def _fake(*groups):
            raise RuntimeError("f_oneway blew up")

    monkeypatch.setattr(mod.stats, "f_oneway", _fake)

    values = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 4) + np.repeat([0.0, 3.0], 10))
    groups = pd.Series(np.repeat(["a", "b"], 10))
    eta, pval = mod._correlation_ratio_with_pvalue(values, groups)

    assert math.isnan(pval), f"pval={pval} was published for an ANOVA that did not run"
    assert math.isfinite(eta) and eta > 0.0, (
        f"eta={eta}: a failing ANOVA must not discard the correlation ratio, which was "
        f"computed from the data before the test was attempted"
    )


@pytest.mark.parametrize("mod", _ETA_MODULES, ids=_ETA_IDS)
def test_control_the_same_partition_gives_a_real_p_value_when_the_anova_runs(mod):
    """Without the monkeypatch the identical fixture must produce a finite,
    significant p. This is what proves the test above pins the branch rather
    than a partition that could never be tested."""
    values = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 4) + np.repeat([0.0, 3.0], 10))
    groups = pd.Series(np.repeat(["a", "b"], 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta, pval = mod._correlation_ratio_with_pvalue(values, groups)

    assert math.isfinite(eta) and eta > 0.5
    assert math.isfinite(pval) and pval < 0.05


# ---------------------------------------------------------------------------
# 4. compute_feature_correlations, the public entry (items 4 and 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "expected_city_eta"), [(0, 0.095829), (7, 0.312773)], ids=["seed0", "seed7"]
)
def test_the_matrix_refuses_the_id_column_and_measures_the_real_one(seed, expected_city_eta):
    """PUBLIC ENTRY. Both halves of the same call, in one assertion block.

    ``customer_id`` (60 singleton levels over 60 rows) must be nan; ``city``
    (3 levels) must be the MEASURED value, and the two seeds must differ,
    which is what tells a reader a measurement happened at all.

    ITEM 6. tests/test_bgl_stage2_s2g07.py stated 0.334933 / 0.134669 as the
    measured control values for this exact fixture. Re-measured 2026-09-17 they
    are 0.095829 / 0.312773; the assertion here is tight enough that a drift in
    either direction fails, rather than the loose ``0 <= cell < 1`` that let the
    wrong numbers stand in prose for as long as they did.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(_s2g07_frame(seed), protected_attributes=["age"])

    assert not np.isfinite(matrix.correlations.loc["customer_id", "age"])
    assert matrix.correlations.loc["city", "age"] == pytest.approx(expected_city_eta, abs=5e-6)


def test_control_a_strong_continuous_categorical_association_still_reads_strong():
    """The guard must not flatten a genuine proxy in the matrix either."""
    rng = np.random.default_rng(11)
    city = np.resize(np.array(["A", "B", "C"]), 90)
    offsets = {"A": 0.0, "B": 20.0, "C": 40.0}
    df = pd.DataFrame(
        {"age": np.array([offsets[c] for c in city]) + rng.normal(0, 1.0, 90), "city": city}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(df, protected_attributes=["age"])

    assert matrix.correlations.loc["city", "age"] > 0.9
    assert matrix.pvalues.loc["city", "age"] < 1e-20


def test_cramers_v_refuses_a_table_the_bias_correction_has_consumed(monkeypatch):
    """ITEM 5, found ALREADY CORRECT and pinned. A 60x60 table over 60 rows
    leaves ``min_corr == 0``: there is nothing left to divide by. That is a
    could-not-check, and 0.0 would be the weakest association on the scale,
    which grades NEGLIGIBLE. The builtin ``max(0, nan) == 0`` idiom this module
    bans by name is pinned in the same breath.
    """
    x = pd.Series([f"z{i}" for i in range(60)])
    y = pd.Series([f"r{i}" for i in range(60)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v, pval = fe_corr._cramers_v_with_pvalue(x, y)

    assert math.isnan(v) and math.isnan(pval)
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught)

    # (b): a non-finite chi-square must propagate, not clamp to a clean zero.
    def _nan_chi2(table):
        return float("nan"), float("nan"), 1, None

    monkeypatch.setattr(fe_corr.stats, "chi2_contingency", _nan_chi2)
    v2, _ = fe_corr._cramers_v_with_pvalue(pd.Series(["a", "b"] * 30), pd.Series(["x", "y"] * 30))
    assert math.isnan(v2), f"a chi-square of nan produced a measured V of {v2}"


def test_control_cramers_v_still_measures_a_real_categorical_association():
    """OVER-CORRECTION CONTROL: two genuinely associated 2-level columns."""
    a = pd.Series(["p"] * 50 + ["q"] * 50)
    b = pd.Series(["x"] * 45 + ["y"] * 5 + ["y"] * 45 + ["x"] * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, pval = fe_corr._cramers_v_with_pvalue(a, b)

    assert math.isfinite(v) and v > 0.5
    assert math.isfinite(pval) and pval < 0.001


# ---------------------------------------------------------------------------
# 5. identify_proxy_variables: a refused headline is not a measured zero (item 3)
# ---------------------------------------------------------------------------


def _unmeasurable_eta_frame(n: int = 120) -> pd.DataFrame:
    """61 'race' levels over 120 rows: eta is refused by branch (b), while
    mutual information and Pearson on category codes still compute."""
    rng = np.random.default_rng(0)
    race = [f"r{i}" for i in range(59) for _ in range(2)] + ["s1", "s2"]
    return pd.DataFrame({"race": race, "age": rng.normal(45, 12, n)})


def test_a_refused_headline_association_is_nan_and_says_so_where_a_reader_looks():
    """PUBLIC ENTRY, ``feature_engineering.correlation.identify_proxy_variables``.

    BEFORE (audit, item 3): ``correlation_results.get("correlation_ratio", 0)``
    turned the refusal into correlation=0, published as NEGLIGIBLE beside a
    measured mutual information.

    AFTER: correlation is nan, ``measures_not_computed`` names it,
    ``headline_measure_status`` is not_assessed, the first recommendation says
    the pair was NOT ASSESSED, and the row is still RETURNED so its other
    evidence survives. A deleted row is a finding a reader can never get back.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = fe_corr.identify_proxy_variables(
            _unmeasurable_eta_frame(),
            ["race"],
            correlation_threshold=0.0,
            significance_level=1.0,
        )

    assert len(results) == 1, "the row must survive the refusal, carrying its other evidence"
    row = results[0]
    assert math.isnan(row.correlation), f"correlation={row.correlation} reads as a measurement"
    measures = row.evidence["correlation_measures"]
    assert "correlation_ratio" in measures["measures_not_computed"]
    assert measures["headline_measure_status"] == "not_assessed"
    assert row.recommendations and row.recommendations[0].startswith("NOT ASSESSED:")
    assert row.evidence["pvalue_status"] == "could_not_compute"
    # The measures that DID compute are still published: refusing the headline
    # must not throw away the rest of the evidence.
    assert math.isfinite(measures["mutual_information"])
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught)


def test_control_a_measured_headline_carries_no_not_assessed_disclosure():
    """OVER-CORRECTION CONTROL. On the genuine proxy frame the headline IS
    measured, so none of the could-not-check machinery may appear: no
    measures_not_computed entry for eta, no NOT ASSESSED recommendation, and a
    risk level that is actually earned."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fe_corr.identify_proxy_variables(
            _genuine_proxy_frame(), ["race"], min_sample_size=100
        )

    assert len(results) == 1
    row = results[0]
    assert math.isfinite(row.correlation) and row.correlation > 0.99
    assert row.risk_level is fe_corr.ProxyRiskLevel.CRITICAL
    measures = row.evidence["correlation_measures"]
    assert "correlation_ratio" not in measures.get("measures_not_computed", [])
    assert "headline_measure_status" not in measures
    assert not any(r.startswith("NOT ASSESSED:") for r in row.recommendations)


# ---------------------------------------------------------------------------
# 6. power_analysis: calculate_intersectional_power counted ROWS (DEFECT, item 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["none", "five"], ids=["no_observation", "five_of_120"])
def test_power_is_not_computed_from_rows_that_carry_no_outcome(outcome):
    """PUBLIC ENTRY, ``FairnessExperiment.calculate_intersectional_power``.

    BEFORE (measured 2026-09-17): three frames identical in shape, 60 rows per
    intersection per arm, differing only in how many outcome values existed
    (240 real / 5 real / 0 real), all returned ``{('f',): 0.7818, ('m',):
    0.7818}`` with NO warning, because the method counted ``len(slice)``.
    ``run_full_analysis()`` then wrote 0.7818 into
    ``ExperimentResult.power_results``.

    AFTER: nan for both intersections, with a warning naming rows AND
    observations.
    """
    exp = FairnessExperiment(_arm(60, 0, outcome), _arm(60, 1, outcome), ["gender"], "y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        power = exp.calculate_intersectional_power(effect_size=0.5)

    assert all(math.isnan(v) for v in power.values()), (
        f"power was reported from rows that carry no outcome: {power}"
    )
    assert any("NO outcome observation" in str(w.message) for w in caught)
    assert any("rows carry" in str(w.message) for w in caught), (
        "the warning must state both numbers: rows alone hides the problem"
    )


def test_run_full_analysis_does_not_publish_a_row_derived_power():
    """The same defect one layer up: the value reaches ExperimentResult."""
    exp = FairnessExperiment(_arm(60, 0, "none"), _arm(60, 1, "none"), ["gender"], "y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = exp.run_full_analysis()

    assert result.power_results, "the key must still be present, carrying its refusal"
    assert all(math.isnan(v) for v in result.power_results.values()), (
        f"ExperimentResult.power_results published {result.power_results}"
    )


def test_control_a_fully_observed_experiment_still_reports_the_same_power():
    """OVER-CORRECTION CONTROL, and the number that makes the defect visible.

    0.7818 is the correct answer for 60 observations per arm at d=0.5, and it
    is the SAME number the broken code returned for an arm holding nothing.
    Healthy data must be untouched by the fix, and must raise no warning."""
    exp = FairnessExperiment(_arm(60, 0, "real"), _arm(60, 1, "real"), ["gender"], "y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        power = exp.calculate_intersectional_power(effect_size=0.5)

    assert power == {("f",): 0.7818, ("m",): 0.7818}
    assert not [w for w in caught if "outcome observation" in str(w.message)]


# ---------------------------------------------------------------------------
# 7. power_analysis: the three already-correct dispositions (items 1, 2, 4)
# ---------------------------------------------------------------------------


def _analyzer(control: pd.DataFrame, treatment: pd.DataFrame) -> FairnessPowerAnalyzer:
    return FairnessPowerAnalyzer(
        FairnessExperiment(control, treatment, ["gender"], "y"), PowerConfig()
    )


def test_an_arm_with_no_rows_at_all_is_still_a_sample_size_shortfall():
    """ITEM 1, found ALREADY CORRECT and pinned. ZERO ROWS is a MEASURED zero:
    we know there are none, and collecting them is exactly the fix. A guard
    that folds it into "rows are present, collecting more will not help"
    DELETES a correct finding (+63 samples, counted in n_underpowered) and
    demotes it to not_assessed. Only "rows present, no outcome value" earns
    that rationale."""
    control = _arm(60, 0, "real")
    treatment = _arm(60, 1, "real")
    treatment = treatment[treatment.gender != "f"].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = _analyzer(control, treatment).adaptive_sampling_plan(budget=100, effect_size=0.5)

    assert plan.allocations.get(("f",)) == 63, (
        f"the no-rows arm lost its allocation: {plan.allocations}"
    )
    assert ("f",) not in plan.not_assessed
    assert plan.n_underpowered == 2
    assert "collecting more rows will not help" not in plan.rationale[("f",)], (
        "that rationale is false about an arm that has no rows"
    )


def test_control_rows_present_but_unrecorded_is_still_not_assessed():
    """THE OTHER SIDE of item 1. 60 rows carrying 0 outcome values is NOT a
    sample-size shortfall, and must keep the not_assessed disposition. If this
    and the test above are not both green, the condition has been widened or
    narrowed past its justification."""
    control = _arm(60, 0, "real")
    treatment = _arm(60, 1, "real")
    treatment.loc[treatment.gender == "f", "y"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = _analyzer(control, treatment).adaptive_sampling_plan(budget=100, effect_size=0.5)

    assert ("f",) in plan.not_assessed
    assert ("f",) not in plan.allocations
    assert "collecting more rows will not help" in plan.rationale[("f",)]


def test_a_sequential_test_with_no_observations_could_not_check():
    """ITEM 2, found ALREADY CORRECT and pinned. 0.0 is a value on the LLR
    scale, and it is the exact midpoint of the Wald boundaries: "the evidence
    so far is perfectly balanced", for a test that was never run."""
    control = _arm(60, 0, "real")
    treatment = _arm(60, 1, "real")
    treatment.loc[treatment.gender == "f", "y"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = _analyzer(control, treatment).sequential_test(effect_size=0.5)

    empty = results[("f",)]
    assert empty.decision is SPRTDecision.COULD_NOT_CHECK
    assert math.isnan(empty.log_likelihood_ratio)
    assert empty.stopped_early is False
    assert any("no likelihood ratio exists" in str(w.message) for w in caught)

    # CONTROL, same call: the observed arm still takes a real decision on a
    # real LLR. A refusal that spreads to the healthy intersection would be
    # indistinguishable in the first two assertions alone.
    observed = results[("m",)]
    assert observed.decision is not SPRTDecision.COULD_NOT_CHECK
    assert math.isfinite(observed.log_likelihood_ratio)


def test_power_result_discloses_rows_and_observations_separately():
    """ITEM 4, found ALREADY CORRECT and pinned. ``n_control`` silently changed
    meaning from rows to observations, so a reader comparing two runs could not
    tell whether a drop meant lost rows or lost outcome values. Both numbers
    must reach ``get_detailed_results``/``PowerResult.to_dict``, not only the
    summary DataFrame."""
    control = _arm(60, 0, "real")
    treatment = _arm(60, 1, "real")
    treatment.loc[treatment.gender == "f", "y"] = np.nan
    analyzer = _analyzer(control, treatment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detailed = {r.intersection: r for r in analyzer.get_detailed_results(effect_size=0.5)}
        summary = analyzer.get_power_summary(effect_size=0.5)

    unobserved = detailed[("f",)]
    assert (unobserved.n_control, unobserved.n_treatment) == (60, 0)
    assert (unobserved.n_rows_control, unobserved.n_rows_treatment) == (60, 60)
    assert math.isnan(unobserved.power)
    assert unobserved.is_powered is None, "not computed is not the same as underpowered"
    as_dict = unobserved.to_dict()
    assert as_dict["n_rows_control"] == 60 and as_dict["n_rows_treatment"] == 60

    # CONTROL: the observed intersection is graded, and its two nouns agree.
    observed = detailed[("m",)]
    assert (observed.n_control, observed.n_treatment) == (60, 60)
    assert observed.power == pytest.approx(0.7818, abs=1e-4)
    assert observed.is_powered is False

    assert set(summary.columns) >= {
        "n_control",
        "n_treatment",
        "n_rows_control",
        "n_rows_treatment",
    }
