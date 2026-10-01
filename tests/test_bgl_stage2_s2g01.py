"""Beta Go-Live Stage 2, group s2g01: twelve could-not-checks reported as measurements.

Every test here pins a THREE-STATE answer (measured / failed / could-not-check) at
the PUBLIC entry point named in the finding, and every one is paired with a CONTROL
proving healthy data still produces the real measurement. A fix that makes
everything refuse is a worse defect than the one it replaces, and it passes any
test that only exercises the degenerate case.

The findings, in file order:

  _statistics.py
    S2G01-1  odds_ratio decided the direction of a degenerate 2x2 table from ONE
             arm, so odds_ratio(30,30,30,30) -> inf: two IDENTICAL groups reported
             as "infinitely higher odds in group 1", while odds_ratio(29,30,29,30)
             -> 1.0. The value jumped to infinity as the groups became MORE equal.
             risk_ratio, in the same dict, published a zero-width CI (1.0, 1.0, 1.0).
    S2G01-2  bayesian_proportion_ci(0, 0) returned the PRIOR MEAN 0.5 as
             point_estimate with a [0.025, 0.975] interval and is_significant=True.
             It also accepted successes > trials (11 of 4 -> point_estimate 2.0).

  attribution.py
    S2G01-3  FeatureAttributionExplainer.global_importance on a CONSTANT target:
             sklearn's r2_score substitutes 0.0 for an undefined R^2 without
             warning, so every feature read importance 0.0 / 'neutral' / notes [].

  classification.py
    S2G01-4  conditional_adverse_impact returned 0.0 ("no adverse impact") for an
             all-NaN decision column, because np.unique collapses [nan...] to one
             element and the CONSTANT-decision guard cannot tell missing from
             constant. Partial missingness was refused loudly; total missingness
             produced the cleanest number the metric can make.
    S2G01-5  conditional_demographic_disparity_with_ci (the EU-hiring SEALED
             primary) hard-coded the percentile bootstrap interval, which this
             repo's own stratified_bootstrap_ci docstring records as having 0%
             coverage at the parity null for a folded spread.

  counterfactual_metric.py
    S2G01-6  counterfactual_fairness scored NaN rows as decision 0 on BOTH sides:
             200 rows of pure NaN returned flip_rate 0.0, severity 'info' and an
             interpretation affirming robustness; half-NaN halved a real 0.60 rate
             to 0.30 while n still claimed all 200 rows.

  fairness_decomposition.py
    S2G01-7  _abs_corr had no sample-size floor (at n=2 every |corr| is 1.0 BY
             CONSTRUCTION and f0 was named in prose as a proxy) and returned 0.0,
             "definitely not a proxy", for constant and all-NaN columns.

  intersectional.py
    S2G01-8  identify_privileged_groups reported false_positive_rate 0.0 -- the
             BEST possible FPR -- for a cell with no actual negatives, and the two
             ratios fell back to 1.0, "exactly at parity", on an empty denominator.
    S2G01-9  intersectional_disparity_analysis claimed intersectional_reveals_more
             False and hidden_disparity 0.0 for a comparison nobody ran.

  robustness.py
    S2G01-10 permutation_test took min_attainable_p from the number of DRAWS, never
             the number of distinct label arrangements, so a 2/2 split of 4 rows
             (6 arrangements, true floor ~0.33) advertised a floor of 9.999e-05 and
             detectable_at_05=True.
    S2G01-11 sensitivity_analysis: int(n * rate) TRUNCATES, so n * rate < 1 flipped
             nothing in any draw and reported is_robust=True, robustness_score=1.0
             from n_iterations no-ops. Reached on healthy data at n=90, rate=1%.
    S2G01-12 stress_test_fairness counted those no-ops as tests that RAN, and
             published worst_case_deviation 0.0 when nothing was assessed.

Written for BGL Stage 2; a new file on purpose, because thirteen other agents are
working in this checkout.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _caught(fn, *args, **kwargs):
    """Run fn, returning (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


def _spread(y_pred, attr):
    """max-min selection rate: a real fairness metric for the robustness tests."""
    y_pred = np.asarray(y_pred, dtype=float)
    attr = np.asarray(attr)
    rates = [y_pred[attr == g].mean() for g in np.unique(attr)]
    return float(max(rates) - min(rates)) if len(rates) > 1 else float("nan")


# ---------------------------------------------------------------------------
# S2G01-1  odds_ratio / risk_ratio on a degenerate table
# ---------------------------------------------------------------------------


def test_two_identical_saturated_groups_are_not_an_infinite_odds_ratio():
    """REFUSAL PIN. Before: odds_ratio(30,30,30,30) -> (inf, nan, nan), and
    compute_effect_sizes at the public entry put that inf in the 'odds_ratio'
    slot for two groups with an IDENTICAL selection rate of 1.0."""
    from vfairness.evaluation.vfairness_metrics import compute_effect_sizes, odds_ratio

    # Both arms saturated: the odds ratio is inf/inf, which is undefined.
    value, caught = _caught(odds_ratio, 30, 30, 30, 30)
    assert math.isnan(value[0]), f"odds_ratio(30,30,30,30) returned {value[0]!r}"
    assert caught, "a could-not-check returned no warning"
    # The other 0/0 table: nobody selected in either group.
    assert math.isnan(_caught(odds_ratio, 0, 30, 0, 30)[0][0])

    # ...and the same through the public entry the finding names.
    groups = np.array(["a"] * 30 + ["b"] * 30)
    ones = np.ones(60, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(ones, ones, groups)["a_vs_b"]
    assert math.isnan(effects["odds_ratio"][0]), (
        f"identical groups reported odds_ratio {effects['odds_ratio'][0]!r}"
    )


def test_a_one_sided_degenerate_table_is_still_a_real_infinity():
    """CONTROL for the refusal above. Only ONE arm degenerate is a genuine
    infinite (or zero) odds ratio and must survive the fix."""
    from vfairness.evaluation.vfairness_metrics import odds_ratio

    # Group 1 saturated, group 2 not: a genuinely infinite odds in group 1.
    assert math.isinf(odds_ratio(30, 30, 15, 30)[0])
    # Group 1 has no events, group 2 does: a real 0.0.
    assert odds_ratio(0, 30, 15, 30)[0] == 0.0
    # And an ordinary table is unchanged.
    assert odds_ratio(40, 100, 20, 100)[0] == pytest.approx(40 / 60 / (20 / 80))


def test_a_saturated_risk_ratio_does_not_publish_a_zero_width_interval():
    """REFUSAL PIN. Before: risk_ratio(30,30,30,30) -> (1.0, 1.0, 1.0), a 95%
    interval of zero width, i.e. perfect certainty, from a table with none."""
    from vfairness.evaluation.vfairness_metrics import risk_ratio

    (rr, lower, upper), caught = _caught(risk_ratio, 30, 30, 30, 30)
    assert rr == 1.0, "the point estimate is measurable and must stay"
    assert math.isnan(lower) and math.isnan(upper)
    assert caught


def test_a_real_risk_ratio_keeps_its_interval():
    """CONTROL: an ordinary table still gets a real, two-sided interval."""
    from vfairness.evaluation.vfairness_metrics import risk_ratio

    rr, lower, upper = risk_ratio(40, 100, 20, 100)
    assert rr == pytest.approx(2.0)
    assert lower < rr < upper
    assert upper - lower > 0.0


def test_an_unmeasurable_effect_size_is_not_graded_large():
    """compute_effect_sizes graded `abs(d) < 0.2` ladders; every comparison is
    False for NaN, so a NaN effect size fell through to 'large effect'."""
    from vfairness.evaluation.vfairness_metrics.classification import compute_effect_sizes

    # A measured, genuinely negligible effect keeps its band (the control half).
    groups = np.array(["a"] * 30 + ["b"] * 30)
    ones = np.ones(60, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(ones, ones, groups)["a_vs_b"]
    assert effects["interpretation"] == "negligible effect"

    # And a real, large one is still called large.
    y_pred = np.concatenate([np.ones(30, dtype=int), np.zeros(30, dtype=int)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big = compute_effect_sizes(y_pred, y_pred, groups)["a_vs_b"]
    assert big["interpretation"] == "large effect"


# ---------------------------------------------------------------------------
# S2G01-2  bayesian_proportion_ci with nothing observed
# ---------------------------------------------------------------------------


def test_zero_trials_does_not_report_the_prior_mean_as_a_measurement():
    """REFUSAL PIN. Before: point_estimate 0.5, [0.025, 0.975], is_significant
    True, from a posterior that IS the uniform prior."""
    import vfairness

    result, caught = _caught(vfairness.bayesian_proportion_ci, 0, 0)
    assert math.isnan(result.point_estimate), f"got {result.point_estimate!r}"
    assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
    assert result.is_significant is False, "an unobserved proportion was graded significant"
    assert result.sample_size == 0
    assert result.metadata.get("could_not_check") is True
    assert caught and "prior" in caught[0]

    # The refusal is visible in the serialised payload too, not only in a warning.
    payload = result.to_dict()
    assert math.isnan(payload["point_estimate"])
    assert payload["is_significant"] is False


def test_an_impossible_count_pair_is_refused_not_scored():
    """Before: bayesian_proportion_ci(11, 4) returned point_estimate 2.0, a
    'proportion' above 1, from a posterior with a negative beta parameter."""
    import vfairness
    from vfairness.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        vfairness.bayesian_proportion_ci(11, 4)
    with pytest.raises(ConfigurationError):
        vfairness.bayesian_proportion_ci(-1, 10)


def test_a_real_proportion_still_gets_its_posterior():
    """CONTROL: observed data still produces the Beta-Binomial estimate."""
    import vfairness

    result = vfairness.bayesian_proportion_ci(30, 100)
    assert result.point_estimate == pytest.approx(31 / 102)
    assert result.lower_bound < result.point_estimate < result.upper_bound
    assert result.sample_size == 100
    assert result.metadata.get("could_not_check") is None
    # Zero successes out of real trials is a MEASUREMENT, not a refusal.
    assert vfairness.bayesian_proportion_ci(0, 50).point_estimate == pytest.approx(1 / 52)


def test_the_group_metrics_route_does_not_keep_the_priors_interval():
    """Second public route from the finding: a group with no true positives
    NaN'd its point estimate but kept the prior's [0.025, 0.975] and
    is_significant=True."""
    from vfairness import get_group_metrics_with_ci

    y_true = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    y_pred = np.concatenate([np.ones(30, dtype=int), np.zeros(30, dtype=int)])
    sensitive = np.array(["A"] * 30 + ["B"] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_group_metrics_with_ci(
            y_true, y_pred, sensitive, min_group_size=0, method="bayesian"
        )
    tpr = result["A"]["tpr"]  # group A has zero actual positives
    assert math.isnan(tpr.point_estimate)
    assert math.isnan(tpr.lower_bound) and math.isnan(tpr.upper_bound)
    assert tpr.is_significant is False


# ---------------------------------------------------------------------------
# S2G01-3  permutation importance against a constant target
# ---------------------------------------------------------------------------


def _weighted_model():
    from vfairness.evaluation.vfairness_metrics.attribution import FeatureAttributionExplainer

    weights = np.array([2.0, 0.5, 0.1])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    explainer = FeatureAttributionExplainer(
        predict=lambda A: A @ weights, feature_names=["f0", "f1", "f2"]
    )
    return explainer, X, weights


def test_a_constant_target_is_not_three_neutral_features():
    """REFUSAL PIN. R^2 is undefined on a constant target; sklearn substitutes
    0.0 silently, so every mean_drop was 0.0 and the result declared f0/f1/f2
    exactly neutral for a model whose first feature carries weight 2.0."""
    explainer, X, _ = _weighted_model()

    result, caught = _caught(explainer.global_importance, X, y=np.ones(200))

    assert all(math.isnan(c.importance) for c in result.contributions), (
        f"got {[c.importance for c in result.contributions]}"
    )
    assert all(c.direction == "not_assessed" for c in result.contributions)
    assert result.notes, "the notes field exists for exactly this disclosure and was empty"
    assert "constant target" in result.notes[0]
    assert caught

    # The fixture must exercise the SCORED path, not the label-free fallback.
    assert result.method == "permutation"


def test_the_constant_target_guard_is_about_variation_not_row_count():
    """The guard must fire on y = ones(60) -- plenty of rows, no variation --
    not merely on a short array."""
    explainer, X, _ = _weighted_model()
    result, _ = _caught(explainer.global_importance, X[:60], y=np.ones(60))
    assert all(math.isnan(c.importance) for c in result.contributions)


def test_a_real_target_still_recovers_the_true_weights():
    """CONTROL: the same model and X with a real target still ranks f0 > f1 > f2."""
    explainer, X, weights = _weighted_model()
    result = explainer.global_importance(X, y=X @ weights)
    by_name = {c.feature: c.importance for c in result.contributions}
    assert all(math.isfinite(v) for v in by_name.values())
    assert by_name["f0"] > by_name["f1"] > by_name["f2"]
    assert by_name["f0"] > 1.0
    assert result.notes == []


# ---------------------------------------------------------------------------
# S2G01-4  conditional_adverse_impact on decisions nobody recorded
# ---------------------------------------------------------------------------


def _adverse_impact_fixture(seed=11, n=2000):
    rng = np.random.default_rng(seed)
    attr = rng.choice(["F", "M"], n)
    covariates = rng.normal(0, 1, n).reshape(-1, 1)
    y_pred = (rng.random(n) < np.where(attr == "M", 0.686, 0.252)).astype(float)
    return y_pred, attr, covariates


def test_decisions_nobody_recorded_are_not_an_absence_of_adverse_impact():
    """REFUSAL PIN. np.unique on an all-NaN column collapses to ONE element, so
    the constant-decision guard returned the literal 0.0 for 2000 rows in which
    no selection decision was recorded at all."""
    import vfairness

    _y, attr, covariates = _adverse_impact_fixture()
    value, caught = _caught(
        vfairness.conditional_adverse_impact, np.full(len(attr), np.nan), attr, covariates
    )
    assert math.isnan(value), f"all-NaN decisions returned {value!r}"
    assert caught and "not finite" in caught[0]

    # Partial missingness refuses the same way instead of raising out of sklearn.
    y_pred, attr, covariates = _adverse_impact_fixture()
    partial = y_pred.copy()
    partial[:800] = np.nan
    partial_value, partial_caught = _caught(
        vfairness.conditional_adverse_impact, partial, attr, covariates
    )
    assert math.isnan(partial_value)
    assert partial_caught


def test_a_constant_recorded_decision_still_reads_zero_adverse_impact():
    """CONTROL, and the finding Stage 1 warns about deleting: a genuinely
    CONSTANT decision is a measured absence of adverse impact and must keep its
    0.0. Only MISSING is refused."""
    import vfairness

    _y, attr, covariates = _adverse_impact_fixture()
    assert vfairness.conditional_adverse_impact(np.ones(len(attr)), attr, covariates) == 0.0


def test_a_real_adjusted_gap_is_still_measured():
    """CONTROL: recorded decisions on the same rows still produce the real gap."""
    import vfairness

    y_pred, attr, covariates = _adverse_impact_fixture()
    value = vfairness.conditional_adverse_impact(y_pred, attr, covariates)
    assert math.isfinite(value)
    assert value > 0.3


def test_covariates_nobody_recorded_refuse_instead_of_raising_linalgerror():
    """The covariate arm of the same defect: an all-NaN covariate raised
    LinAlgError('SVD did not converge') out of the identifiability guard."""
    import vfairness

    y_pred, attr, _cov = _adverse_impact_fixture()
    value, caught = _caught(
        vfairness.conditional_adverse_impact, y_pred, attr, np.full((len(attr), 1), np.nan)
    )
    assert math.isnan(value)
    assert caught and "covariate" in caught[0]


# ---------------------------------------------------------------------------
# S2G01-5  the sealed CDD interval at the parity null
# ---------------------------------------------------------------------------


def _cdd_at_exact_parity(rng, per_cell=100):
    """Identical selection probability in BOTH groups within every stratum, so
    the true conditional demographic disparity is exactly 0.0."""
    y_pred, attr, strata = [], [], []
    for stratum, p in {0: 0.25, 1: 0.45, 2: 0.60, 3: 0.75}.items():
        for group in ("A", "B"):
            y_pred.append(rng.binomial(1, p, per_cell))
            attr.append(np.full(per_cell, group))
            strata.append(np.full(per_cell, stratum))
    return np.concatenate(y_pred), np.concatenate(attr), np.concatenate(strata)


def test_the_sealed_cdd_interval_covers_the_parity_null():
    """REFUSAL PIN, measured as COVERAGE rather than as one number. The seal
    gates on this CI's upper bound. With the hard-coded percentile interval the
    95% CI never touched zero for a model at EXACT parity (0 of 40 replications
    covered the truth); the basic interval, which this repo's own
    stratified_bootstrap_ci docstring prescribes for a folded spread, covers it.
    """
    from vfairness import conditional_demographic_disparity_with_ci as cdd_ci

    reps, covered = 20, 0
    for k in range(reps):
        rng = np.random.default_rng(1000 + k)
        y_pred, attr, strata = _cdd_at_exact_parity(rng)
        result = cdd_ci(y_pred, attr, strata, n_bootstrap=400, random_state=k)
        if result.lower_bound <= 0.0 <= result.upper_bound:
            covered += 1
    # Nominal 95%; measured 37/40 on the default. Pinned well below that so the
    # test is about the ESTIMATOR, not about bootstrap noise -- the percentile
    # default scored 0/40 and cannot reach this bar.
    assert covered >= int(0.80 * reps), f"only {covered}/{reps} intervals covered true parity"


def test_the_percentile_interval_is_still_reachable_for_comparison():
    """The default changed; the old construction stays available so a sealed
    cell issued before the change can be reproduced."""
    from vfairness import conditional_demographic_disparity_with_ci as cdd_ci

    rng = np.random.default_rng(207)
    y_pred, attr, strata = _cdd_at_exact_parity(rng, per_cell=50)
    result = cdd_ci(y_pred, attr, strata, n_bootstrap=400, random_state=1, method="percentile")
    assert result.metadata["interval_construction"] == "percentile"


def test_a_real_disparity_is_still_significant_under_the_new_interval():
    """CONTROL. Switching the interval must not cost the power to see a real
    gap: a planted 0.20 within-stratum disparity still excludes zero."""
    from vfairness import conditional_demographic_disparity_with_ci as cdd_ci

    rng = np.random.default_rng(5)
    y_pred, attr, strata = [], [], []
    for stratum, p in {0: 0.25, 1: 0.45, 2: 0.60, 3: 0.75}.items():
        for group, delta in (("A", 0.0), ("B", 0.20)):
            y_pred.append(rng.binomial(1, min(p + delta, 0.95), 300))
            attr.append(np.full(300, group))
            strata.append(np.full(300, stratum))
    result = cdd_ci(
        np.concatenate(y_pred),
        np.concatenate(attr),
        np.concatenate(strata),
        n_bootstrap=500,
        random_state=1,
    )
    assert result.point_estimate == pytest.approx(0.19, abs=0.05)
    assert result.lower_bound > 0.0
    assert result.is_significant is True


# ---------------------------------------------------------------------------
# S2G01-6  counterfactual fairness over rows nobody could score
# ---------------------------------------------------------------------------


def test_unreadable_scores_are_not_evidence_of_robustness():
    """REFUSAL PIN. NaN compares False against any threshold, so every
    unreadable row became decision 0 on BOTH sides: 200 rows of pure NaN
    returned flip_rate 0.0, severity 'info', n 200, and an interpretation that
    affirmatively asserts robustness."""
    from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
        counterfactual_fairness,
    )

    result, caught = _caught(
        counterfactual_fairness,
        y_pred_factual=np.full(200, np.nan),
        y_pred_counterfactual=np.full(200, np.nan),
        threshold=0.5,
    )
    assert result.flip_rate is None, f"got flip_rate {result.flip_rate!r}"
    assert result.severity == "not_assessed"
    assert result.n == 0, "n claimed support from rows nobody could score"
    assert "Robust to the perturbation" not in result.interpretation
    assert "could NOT be assessed" in result.interpretation
    assert result.notes
    assert caught


def test_unreadable_rows_no_longer_dilute_a_real_flip_rate():
    """The quieter half: 60 of 100 readable rows genuinely flip (a true rate of
    0.60) and the other 100 rows are NaN. Before, the unreadable rows counted as
    non-flips and halved it to 0.30 while n still said 200."""
    from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
        counterfactual_fairness,
    )

    rng = np.random.default_rng(3)
    factual = rng.random(100)
    counterfactual = factual.copy()
    counterfactual[:60] = 1 - factual[:60]
    true_rate = float(((factual >= 0.5).astype(int) != (counterfactual >= 0.5).astype(int)).mean())

    padded_f = np.concatenate([factual, np.full(100, np.nan)])
    padded_c = np.concatenate([counterfactual, np.full(100, np.nan)])
    result, caught = _caught(counterfactual_fairness, padded_f, padded_c, 0.5)

    assert result.flip_rate == pytest.approx(true_rate)
    assert result.n == 100
    assert caught and "excluded" in caught[0]
    assert result.notes and "could not be scored" in result.notes[0]


def test_label_inputs_do_not_count_an_unreadable_row_as_a_flip():
    """With threshold=None the same rows ran the OTHER way: NaN != NaN is True,
    so every unreadable row counted as a flip."""
    from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
        counterfactual_fairness,
    )

    result, _ = _caught(
        counterfactual_fairness,
        np.full(50, np.nan),
        np.full(50, np.nan),
        None,
    )
    assert result.flip_rate is None
    assert result.severity == "not_assessed"


def test_a_real_flip_rate_is_still_measured_and_graded():
    """CONTROL: fully readable scores still produce the real rate and severity."""
    from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
        counterfactual_fairness,
    )

    rng = np.random.default_rng(3)
    factual = rng.random(200)
    counterfactual = factual.copy()
    counterfactual[:60] = 1 - factual[:60]
    result = counterfactual_fairness(factual, counterfactual, 0.5)
    assert result.flip_rate == pytest.approx(0.30)
    assert result.severity == "critical"
    assert result.n == 200
    assert result.notes == []
    # A genuinely unchanged model still reads as robust.
    fair = counterfactual_fairness(factual, factual.copy(), 0.5)
    assert fair.flip_rate == 0.0
    assert fair.severity == "info"


# ---------------------------------------------------------------------------
# S2G01-7  proxy correlations nobody could measure
# ---------------------------------------------------------------------------


def test_a_correlation_of_one_by_construction_is_not_a_proxy_accusation():
    """REFUSAL PIN. A Pearson correlation through two points is +-1 BY
    CONSTRUCTION, so at n=2 every feature scored 1.0, cleared the 0.30
    threshold, entered flagged_proxies, and was named in prose a human reads."""
    from vfairness.evaluation.vfairness_metrics.fairness_decomposition import (
        fairness_decomposition,
    )

    X = np.array([[0.1, 0.9, 0.42], [0.8, 0.15, 0.99]])
    result, _ = _caught(
        fairness_decomposition,
        lambda Z: np.clip(Z[:, 0] * 0.9 + 0.05, 0, 1),
        X,
        np.array(["a", "b"]),
        feature_names=["f0", "f1", "f2"],
    )

    assert result.n == 2, "the fixture must reach the two-row path"
    assert all(v is None for v in result.proxy_scores.values()), f"got {result.proxy_scores}"
    assert result.flagged_proxies == []
    assert "Proxy features carrying it" not in result.interpretation
    # ...and the prose must not swap one unearned claim for another: "diffuse"
    # is also an all-clear about proxies.
    assert "diffuse" not in result.interpretation
    assert "UNKNOWN, not ruled out" in result.interpretation
    assert any("COULD NOT CHECK the proxy correlation" in n for n in result.notes)


def test_a_constant_and_an_all_nan_feature_are_not_cleared_of_being_proxies():
    """The opposite direction, on HEALTHY 400-row data: both read 0.0, i.e.
    'measured, and definitely not a proxy', for a correlation with a zero
    denominator."""
    from vfairness.evaluation.vfairness_metrics.fairness_decomposition import (
        fairness_decomposition,
    )

    rng = np.random.default_rng(7)
    n = 400
    protected = np.array(["a"] * 200 + ["b"] * 200)
    X = np.column_stack(
        [
            (protected == "b").astype(float) + rng.normal(0, 0.05, n),  # the real proxy
            rng.normal(0, 1, n),  # neutral
            np.full(n, 3.0),  # constant
            np.full(n, np.nan),  # unrecorded
        ]
    )
    result, _ = _caught(
        fairness_decomposition,
        lambda Z: np.clip(0.2 + 0.6 * Z[:, 0], 0, 1),
        X,
        protected,
        feature_names=["proxy", "neutral", "const_feature", "nan_feature"],
        max_rows=n,
    )

    assert result.proxy_scores["const_feature"] is None
    assert result.proxy_scores["nan_feature"] is None
    assert any("const_feature" in n and "nan_feature" in n for n in result.notes)

    # CONTROL, in the same call: the real proxy is still measured and flagged,
    # and the neutral feature is still measured and NOT flagged.
    assert result.proxy_scores["proxy"] > 0.9
    assert result.proxy_scores["neutral"] < 0.30
    assert result.flagged_proxies == ["proxy"]
    assert result.severity == "critical"
    assert result.n_advantaged == 200 and result.n_disadvantaged == 200


# ---------------------------------------------------------------------------
# S2G01-8  identify_privileged_groups: an undefined FPR and empty-denominator ratios
# ---------------------------------------------------------------------------


def test_a_cell_with_no_actual_negatives_has_no_false_positive_rate():
    """REFUSAL PIN. Group A has ZERO actual negatives, so its FPR denominator is
    empty; it was reported as a measured 0.0, the BEST possible FPR, next to
    group B's real 0.8."""
    from vfairness import identify_privileged_groups

    sensitive = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.concatenate([np.ones(100, dtype=int), np.ones(50, dtype=int), np.zeros(50, int)])
    y_pred = np.concatenate(
        [np.ones(100, dtype=int), np.ones(50, dtype=int), np.array([1] * 40 + [0] * 10)]
    )
    result, caught = _caught(identify_privileged_groups, y_true, y_pred, sensitive)
    cells = {g.group: g for g in result["all_groups"]}

    assert math.isnan(cells["A"].false_positive_rate), (
        f"got {cells['A'].false_positive_rate!r} for a cell with no actual negatives"
    )
    assert "false_positive_rate" in cells["A"].unmeasured
    assert caught, "the undefined FPR was disclosed nowhere"

    # CONTROL in the same call: B's FPR is real and must stay a number.
    assert cells["B"].false_positive_rate == pytest.approx(0.8)
    assert cells["B"].unmeasured == {}
    assert math.isnan(cells["A"].to_dict()["false_positive_rate"])


def test_an_empty_ratio_denominator_is_not_exact_parity():
    """With nobody selected at all, both ratios fell back to 1.0 -- 'exactly at
    parity' -- for a denominator that was empty."""
    from vfairness import identify_privileged_groups

    sensitive = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    result, _ = _caught(identify_privileged_groups, y_true, np.zeros(200, dtype=int), sensitive)

    for cell in result["all_groups"]:
        assert math.isnan(cell.relative_to_overall), f"{cell.group} claimed parity with nobody"
        assert math.isnan(cell.relative_to_best)
        assert "relative_to_overall" in cell.unmeasured


def test_measured_rates_and_ratios_still_come_back_as_numbers():
    """CONTROL: an ordinary run must not acquire NaNs or a warning."""
    from vfairness import identify_privileged_groups

    rng = np.random.default_rng(0)
    sensitive = np.array(["x"] * 100 + ["y"] * 100)
    y_true = rng.integers(0, 2, 200)
    y_pred = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    result, caught = _caught(identify_privileged_groups, y_true, y_pred, sensitive)

    assert not caught, f"healthy data warned: {caught}"
    for cell in result["all_groups"]:
        assert math.isfinite(cell.false_positive_rate)
        assert math.isfinite(cell.relative_to_overall)
        assert math.isfinite(cell.relative_to_best)
        assert cell.unmeasured == {}
    assert result["max_disparity"] == 1.0
    # A MEASURED zero FPR (real negatives, none wrongly selected) must survive.
    all_negative = np.zeros(200, dtype=int)
    measured, _ = _caught(identify_privileged_groups, all_negative, all_negative, sensitive)
    for cell in measured["all_groups"]:
        assert cell.false_positive_rate == 0.0


# ---------------------------------------------------------------------------
# S2G01-9  a comparison nobody ran
# ---------------------------------------------------------------------------


def test_a_comparison_nobody_ran_reveals_nothing_and_says_so():
    """REFUSAL PIN, on the DOCUMENTED DEFAULT signature and on HEALTHY data: a
    real, measured 100% intersectional gap still reported
    intersectional_reveals_more False, hidden_disparity 0.0 and
    single_attributes_not_assessable [] -- an explicit claim that nothing was
    skipped -- when no single-attribute comparison was performed at all."""
    from vfairness import intersectional_disparity_analysis

    y_true = [1] * 120
    y_pred = [1] * 60 + [0] * 60
    sensitive = ["a_x"] * 60 + ["b_y"] * 60
    result, _ = _caught(intersectional_disparity_analysis, y_true, y_pred, sensitive)
    comparison = result["comparison"]

    assert comparison["intersectional_reveals_more"] is None
    assert math.isnan(comparison["hidden_disparity"])
    assert "comparison_not_run" in comparison
    # CONTROL half, same call: the intersectional disparity itself IS measured
    # and must not have been refused along with the comparison.
    assert comparison["intersectional_disparity"] == 1.0


def test_a_comparison_that_did_run_still_answers():
    """CONTROL: supply single_attributes and the real comparison comes back."""
    from vfairness import intersectional_disparity_analysis

    y_true = [1] * 120
    y_pred = [1] * 60 + [0] * 60
    sensitive = ["a_x"] * 60 + ["b_y"] * 60
    singles = {"g": np.array(["a"] * 60 + ["b"] * 60)}
    result, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, sensitive, single_attributes=singles
    )
    comparison = result["comparison"]

    assert comparison["intersectional_reveals_more"] is False
    assert comparison["hidden_disparity"] == pytest.approx(0.0)
    assert comparison["single_attribute_disparities"] == {"g": 1.0}
    assert "comparison_not_run" not in comparison


# ---------------------------------------------------------------------------
# S2G01-10  the permutation test's design floor
# ---------------------------------------------------------------------------


def test_a_design_that_could_never_fire_does_not_advertise_detectability():
    """REFUSAL PIN at the library's own exported wrapper. Four rows split 2/2
    have C(4,2)=6 distinct label arrangements, so no data can drive this test
    below ~1/3; brute force over all 16 possible y_pred vectors returns
    0.3320667933206679 as the smallest p ANY data produces. The reported floor
    was 9.999e-05 and detectable_at_05 was True."""
    from vfairness import permutation_test_demographic_parity

    sensitive = np.array(["F", "F", "M", "M"])
    # COMPLETE separation: the most extreme data this shape allows.
    result, caught = _caught(
        permutation_test_demographic_parity,
        np.array([1, 1, 0, 0]),
        sensitive,
        n_permutations=2000,
        random_state=0,
    )

    assert result.min_attainable_p_value == pytest.approx(2 / 6)
    assert result.detectable_at_05 is False
    assert "NOT DETECTABLE" in result.design_note
    # The graded negatives were the point: they must become could-not-check.
    assert result.significant_at_05 is None
    assert result.significant_at_01 is None
    assert math.isnan(result.p_value)
    assert caught


def test_the_floor_is_the_larger_of_the_two_limits():
    """The resample floor must still bind when it is the larger one: 10 draws
    over a design with thousands of arrangements cannot beat 1/11."""
    from vfairness import permutation_test_demographic_parity

    rng = np.random.default_rng(0)
    sensitive = np.array(["F"] * 100 + ["M"] * 100)
    y_pred = np.concatenate([rng.binomial(1, 0.3, 100), rng.binomial(1, 0.7, 100)])
    result, _ = _caught(
        permutation_test_demographic_parity,
        y_pred,
        sensitive,
        n_permutations=10,
        random_state=0,
    )
    assert result.min_attainable_p_value == pytest.approx(1 / 11)


def test_a_design_with_real_power_still_reports_a_measured_result():
    """CONTROL: at n=400 with a real 40-point gap the test still fires, the
    floor is the resample floor, and the design note stays empty."""
    from vfairness import permutation_test_demographic_parity

    rng = np.random.default_rng(0)
    sensitive = np.array(["F"] * 200 + ["M"] * 200)
    y_pred = np.concatenate([rng.binomial(1, 0.3, 200), rng.binomial(1, 0.7, 200)])
    result = permutation_test_demographic_parity(
        y_pred, sensitive, n_permutations=2000, random_state=0
    )
    assert math.isfinite(result.p_value)
    assert result.p_value < 0.01
    assert result.significant_at_05 is True
    assert result.detectable_at_05 is True
    assert result.design_note == ""
    assert result.min_attainable_p_value == pytest.approx(1 / 2001)


# ---------------------------------------------------------------------------
# S2G01-11  a perturbation that never happened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perturbation_type", ["label_noise", "group_noise"])
def test_a_zero_flip_budget_is_not_a_robustness_verdict(perturbation_type):
    """REFUSAL PIN on HEALTHY, non-degenerate data: n=90 at the library's own
    default 1% budget flips int(90 * 0.01) = 0 rows, so all 100 draws were the
    unperturbed metric. Before: is_robust True, robustness_score 1.0,
    n_iterations_run 100, n_unmeasurable 0, no warning."""
    import vfairness

    y_pred = np.concatenate([np.ones(45, dtype=int), np.zeros(45, dtype=int)])
    attr = np.array(["A"] * 45 + ["B"] * 45)
    assert int(len(y_pred) * 0.01) == 0, "the fixture must actually reach the zero-flip branch"

    result, caught = _caught(
        vfairness.sensitivity_analysis,
        y_pred,
        attr,
        _spread,
        perturbation_type=perturbation_type,
        perturbation_rate=0.01,
        n_iterations=100,
        random_state=0,
    )
    assert result.is_robust is None, f"got is_robust {result.is_robust!r}"
    assert result.robustness_score is None
    assert math.isnan(result.max_deviation)
    assert result.n_iterations_run == 0, "no-op draws were counted as draws that ran"
    assert caught and "perturbed NOTHING" in caught[0]


def test_the_zero_flip_refusal_is_about_the_budget_not_the_row_count():
    """Small n is not the trigger; n * rate < 1 is. A 2-row set at 60% flips one
    row and IS assessable, while 500 rows at 0.1% flips none."""
    import vfairness

    tiny, _ = _caught(
        vfairness.sensitivity_analysis,
        np.array([1, 0]),
        np.array(["A", "B"]),
        _spread,
        perturbation_type="label_noise",
        perturbation_rate=0.6,
        n_iterations=20,
        random_state=0,
    )
    assert tiny.is_robust is not None, "a budget of 1 flip is a real perturbation"

    big, caught = _caught(
        vfairness.sensitivity_analysis,
        np.concatenate([np.ones(250, dtype=int), np.zeros(250, dtype=int)]),
        np.array(["A"] * 250 + ["B"] * 250),
        _spread,
        perturbation_type="label_noise",
        perturbation_rate=0.001,
        n_iterations=20,
        random_state=0,
    )
    assert big.is_robust is None
    assert caught


def test_a_real_perturbation_still_produces_a_robustness_verdict():
    """CONTROL: n=400 at 5% flips 20 rows and still grades, in both directions."""
    import vfairness

    y_pred = np.concatenate([np.ones(200, dtype=int), np.zeros(200, dtype=int)])
    attr = np.array(["A"] * 200 + ["B"] * 200)
    fragile, caught = _caught(
        vfairness.sensitivity_analysis,
        y_pred,
        attr,
        _spread,
        perturbation_type="label_noise",
        perturbation_rate=0.05,
        n_iterations=50,
        random_state=0,
    )
    assert fragile.is_robust is False
    assert fragile.max_deviation == pytest.approx(0.10, abs=0.02)
    assert fragile.n_iterations_run == 50
    assert not caught

    robust, _ = _caught(
        vfairness.sensitivity_analysis,
        y_pred,
        attr,
        _spread,
        perturbation_type="label_noise",
        perturbation_rate=0.05,
        n_iterations=50,
        robustness_threshold=0.5,
        random_state=0,
    )
    assert robust.is_robust is True


# ---------------------------------------------------------------------------
# S2G01-12  a stress test that counted the no-ops as tests
# ---------------------------------------------------------------------------


def test_no_op_stress_tests_are_not_counted_as_tests_that_ran():
    """REFUSAL PIN on HEALTHY data with the library defaults: at n=90 the 1%
    label_noise and group_noise arms flip int(90 * 0.01) = 0 rows. Before, this
    reported n_tests_run 9, n_tests_not_assessable 0, not_assessable [] and zero
    warnings, while two of those nine perturbed nothing at all."""
    import vfairness

    y_pred = np.concatenate([np.ones(45, dtype=int), np.zeros(45, dtype=int)])
    attr = np.array(["A"] * 45 + ["B"] * 45)
    result, caught = _caught(vfairness.stress_test_fairness, y_pred, attr, _spread, random_state=0)

    assert result["n_tests_not_assessable"] == 2
    assert set(result["not_assessable"]) == {"label_noise @ 1%", "group_noise @ 1%"}
    assert result["n_tests_run"] == 7
    assert caught


def test_a_stress_test_that_assessed_nothing_publishes_no_deviation():
    """At n=2 every arm is unassessable, and worst_case_deviation initialised to
    0.0 published the BEST possible deviation for a test that measured nothing."""
    import vfairness

    result, _ = _caught(
        vfairness.stress_test_fairness,
        np.array([1, 0]),
        np.array(["A", "B"]),
        _spread,
        random_state=0,
    )
    assert result["overall_robust"] is None
    assert result["n_tests_run"] == 0
    assert math.isnan(result["worst_case_deviation"]), f"got {result['worst_case_deviation']!r}"
    assert math.isnan(result["worst_case_metric"])


def test_a_full_stress_test_still_reports_every_arm():
    """CONTROL: at n=400 all nine arms are assessable, the worst case is a real
    measured number, and a measured 0.0 deviation is still reported as 0.0."""
    import vfairness

    y_pred = np.concatenate([np.ones(200, dtype=int), np.zeros(200, dtype=int)])
    attr = np.array(["A"] * 200 + ["B"] * 200)
    result, caught = _caught(vfairness.stress_test_fairness, y_pred, attr, _spread, random_state=0)

    assert result["n_tests_run"] == 9
    assert result["n_tests_not_assessable"] == 0
    assert result["not_assessable"] == []
    assert not caught
    assert result["worst_case_deviation"] == pytest.approx(0.20, abs=0.02)
    assert math.isfinite(result["worst_case_metric"])
    assert result["overall_robust"] is False
    # A measured zero deviation is a measurement and stays 0.0, not NaN.
    subsample = result["perturbation_results"]["subsample"]["1%"]
    assert subsample["max_deviation"] == 0.0
    assert subsample["is_robust"] is True
