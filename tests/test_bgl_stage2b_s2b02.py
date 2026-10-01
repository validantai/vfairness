"""Beta Go-Live stage 2b, group s2b02.

Every test here pins a could-not-check at a PUBLIC entry point, and every pin is
paired with a control proving healthy data still produces the real measurement.
A fix that makes everything refuse passes any test that only checks the
degenerate case, so the controls are the half that matters.

Capabilities covered:
  intersectional_disparity_analysis / identify_privileged_groups
  sensitivity_analysis / stress_test_fairness
  permutation_test / permutation_test_demographic_parity
  permutation_test_equal_opportunity
  create_fairness_loss / equalized_odds_loss
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from vfairness import (
    identify_privileged_groups,
    intersectional_disparity_analysis,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
    sensitivity_analysis,
    stress_test_fairness,
)
from vfairness.evaluation.vfairness_metrics.robustness import (
    _label_arrangement_floor,
    permutation_test,
)


def _caught(fn, *args, **kwargs):
    """Run ``fn`` capturing warnings, returning (result, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# S2B02-01/02  one unmeasurable FPR cell deleted every measured FPR finding
#
# `if group_fpr is None: continue` is False for the NaN this module returns for
# a cell with no actual negatives, so the NaN entered the pooled baseline,
# overall_fpr became NaN, every fpr_delta became NaN, and `nan > 0.05` is False.
# Every fpr_disparity finding in the run disappeared, silently, including fully
# measured critical ones.
# ---------------------------------------------------------------------------


def _three_cells_one_undefined() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A: FPR 0.900 measured. B: FPR 0.033 measured. C: FPR undefined.

    C has 50 rows and every label positive, so fp + tn == 0 and its false
    positive rate has an empty denominator.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    group: List[str] = []
    for i in range(100):  # A: 100 actual negatives, 90 wrongly selected
        y_true.append(0)
        y_pred.append(1 if i < 90 else 0)
        group.append("A")
    for i in range(60):  # B: 60 actual negatives, 2 wrongly selected
        y_true.append(0)
        y_pred.append(1 if i < 2 else 0)
        group.append("B")
    for _ in range(40):  # B: 40 actual positives, all selected
        y_true.append(1)
        y_pred.append(1)
        group.append("B")
    for i in range(50):  # C: 50 actual positives, NO negatives at all
        y_true.append(1)
        y_pred.append(1 if i < 25 else 0)
        group.append("C")
    return np.array(y_true), np.array(y_pred), np.array(group)


def _two_cells_both_measured() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CONTROL fixture: the same A and B, with no undefined cell at all."""
    y_true, y_pred, group = _three_cells_one_undefined()
    keep = group != "C"
    return y_true[keep], y_pred[keep], group[keep]


def _fpr_findings(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {f["groups"][0]: f for f in result["findings"] if f["type"] == "fpr_disparity"}


def test_the_fixture_really_produces_one_undefined_and_two_measured_cells() -> None:
    """FIXTURE ASSERTION. In stage 1 the green sabotage was a fixture that never
    reached the branch, so prove this one does before pinning anything on it."""
    y_true, y_pred, group = _three_cells_one_undefined()
    analysis, _ = _caught(identify_privileged_groups, y_true, y_pred, group, min_group_size=30)

    cells = {g.group: g for g in analysis["all_groups"]}
    assert set(cells) == {"A", "B", "C"}
    assert cells["A"].false_positive_rate == pytest.approx(0.90)
    assert cells["B"].false_positive_rate == pytest.approx(0.0333, abs=1e-4)
    # The branch under test: C's FPR is the third state, not a number.
    assert math.isnan(cells["C"].false_positive_rate)
    assert "false_positive_rate" in cells["C"].unmeasured
    assert cells["A"].unmeasured == {}


def test_one_unmeasurable_cell_does_not_delete_the_measured_fpr_findings() -> None:
    """REFUSAL PIN at the public entry. Before: 2 fpr_disparity findings without
    cell C, 0 with it. After: both survive AND C is named as not compared."""
    y_true, y_pred, group = _three_cells_one_undefined()
    result, messages = _caught(
        intersectional_disparity_analysis, y_true, y_pred, group, min_group_size=30
    )

    found = _fpr_findings(result)
    assert set(found) == {"A", "B"}, "a cell nobody could measure erased the cells that were"
    assert found["A"]["metric_values"]["group_fpr"] == pytest.approx(0.90)
    assert found["B"]["metric_values"]["group_fpr"] == pytest.approx(0.0333, abs=1e-4)
    # The baseline is pooled over the MEASURED cells only, so it is a number.
    assert found["A"]["metric_values"]["overall_fpr"] == pytest.approx(0.575, abs=1e-3)
    assert found["A"]["severity"] == "critical"
    assert found["B"]["severity"] == "critical"

    # ...and the skipped cell is disclosed, so "no finding for C" can never be
    # read as "C was checked and was fine".
    not_assessed = [f for f in result["findings"] if f["type"] == "fpr_not_assessed"]
    assert len(not_assessed) == 1
    entry = not_assessed[0]
    assert entry["groups"] == ["C"]
    assert entry["fpr_comparison_ran"] is True
    assert entry["p_value"] is None, "a could-not-check must not join the correction family"
    assert entry["statistically_significant"] is None
    assert [c["group"] for c in entry["cells_not_compared"]] == ["C"]
    assert "no actual negatives" in entry["cells_not_compared"][0]["reason"]
    assert any("could not be measured for 1 cell(s) (C)" in m for m in messages)


def test_a_finding_that_carries_a_nan_fpr_says_it_is_unmeasured() -> None:
    """The NaN reached findings[].metric_values with nothing beside it saying it
    was never measured; the reason lived only on GroupAdvantage.unmeasured."""
    y_true, y_pred, group = _three_cells_one_undefined()
    result, _ = _caught(intersectional_disparity_analysis, y_true, y_pred, group, min_group_size=30)

    by_group = {
        f["groups"][0]: f
        for f in result["findings"]
        if f["type"] in ("over_prediction", "under_prediction") and f["groups"]
    }
    assert "C" in by_group, "the fixture must produce a prediction finding for cell C"
    c_finding = by_group["C"]
    assert math.isnan(c_finding["metric_values"]["false_positive_rate"])
    assert "false_positive_rate" in c_finding["unmeasured"]
    assert "no actual negatives" in c_finding["unmeasured"]["false_positive_rate"]
    # CONTROL inside the pin: a measured cell's finding claims nothing unmeasured.
    assert by_group["A"]["unmeasured"] == {}
    # Every finding carries the key, so one place is read for the whole list.
    assert all("unmeasured" in f for f in result["findings"])


def test_group_advantage_to_dict_carries_the_unmeasured_map() -> None:
    """Serialisation pin: deleting the key from to_dict() must fail here, or the
    disclosure never reaches a stored audit record."""
    y_true, y_pred, group = _three_cells_one_undefined()
    analysis, _ = _caught(identify_privileged_groups, y_true, y_pred, group, min_group_size=30)
    rows = {g.group: g.to_dict() for g in analysis["all_groups"]}
    assert "unmeasured" in rows["C"]
    assert "false_positive_rate" in rows["C"]["unmeasured"]
    assert rows["A"]["unmeasured"] == {}


def test_healthy_cells_are_still_all_compared_and_nothing_is_disclaimed() -> None:
    """OVER-CORRECTION CONTROL. With no undefined cell the two real findings are
    unchanged and no could-not-check entry is invented."""
    y_true, y_pred, group = _two_cells_both_measured()
    result, messages = _caught(
        intersectional_disparity_analysis, y_true, y_pred, group, min_group_size=30
    )

    found = _fpr_findings(result)
    assert set(found) == {"A", "B"}
    assert found["A"]["metric_values"]["overall_fpr"] == pytest.approx(0.575, abs=1e-3)
    assert not [f for f in result["findings"] if f["type"] == "fpr_not_assessed"]
    assert all(f["unmeasured"] == {} for f in result["findings"])
    assert not any("could not be measured" in m for m in messages)


# ---------------------------------------------------------------------------
# S2B02-03  a perturbation with a budget but no TARGET
#
# 'group_noise' redraws labels from np.unique(sensitive_attr). With one distinct
# value that draw returns the same value every time, so 100 draws at a 10%
# budget are 100 copies of the original and the function graded them
# is_robust=True / robustness_score=1.0 / max_deviation=0.0, with no warning.
# ---------------------------------------------------------------------------


def _dp_spread(y_pred: np.ndarray, attr: np.ndarray) -> float:
    groups = np.unique(attr)
    if groups.size < 2:
        return 0.0
    rates = [float(np.mean(y_pred[attr == g])) for g in groups]
    return max(rates) - min(rates)


def test_group_noise_with_one_group_is_not_a_robustness_pass() -> None:
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 2, 200)
    one_group = np.array(["A"] * 200)
    # FIXTURE ASSERTION: the budget is healthy, so only the missing TARGET can
    # be what refuses. int(200 * 0.10) = 20 rows would have been redrawn.
    assert int(200 * 0.10) == 20
    assert np.unique(one_group).size == 1

    result, messages = _caught(
        sensitivity_analysis,
        y_pred,
        one_group,
        _dp_spread,
        perturbation_type="group_noise",
        perturbation_rate=0.10,
        n_iterations=100,
        random_state=1,
    )
    assert result.is_robust is None, "100 identical no-op draws are not a robustness verdict"
    assert result.robustness_score is None
    assert math.isnan(result.max_deviation)
    assert result.n_iterations_run == 0
    assert any("guaranteed no-op" in m for m in messages)


def test_a_stress_test_says_which_arms_could_not_be_measured() -> None:
    """stress_test_fairness reported 9 of 9 tests run and overall_robust=True on
    the same input. The three group_noise arms must now be named."""
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 2, 200)
    one_group = np.array(["A"] * 200)

    result, messages = _caught(
        stress_test_fairness, y_pred, one_group, _dp_spread, n_iterations=20, random_state=1
    )
    assert result["n_tests_not_assessable"] == 3
    assert all(label.startswith("group_noise") for label in result["not_assessable"])
    assert result["n_tests_run"] == 6
    assert any("could not be measured" in m for m in messages)


def test_a_subsample_that_drops_nothing_is_refused() -> None:
    """PIN for the previously unpinned `subsample_budget >= n` arm: at
    perturbation_rate=0.0 int(n * 1.0) keeps every row."""
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 2, 90)
    attr = np.array(["A"] * 45 + ["B"] * 45)
    # FIXTURE ASSERTION: this is the `>= n` arm, not the `< 1` one.
    assert int(90 * (1 - 0.0)) == 90

    result, messages = _caught(
        sensitivity_analysis,
        y_pred,
        attr,
        _dp_spread,
        perturbation_type="subsample",
        perturbation_rate=0.0,
        n_iterations=10,
        random_state=1,
    )
    assert result.is_robust is None
    assert result.robustness_score is None
    assert any("dropped NOTHING" in m for m in messages)


def test_label_noise_that_complements_to_itself_is_refused() -> None:
    """The label_noise analogue: 1 - y_pred returns y_pred unchanged when every
    prediction is exactly 0.5, so the budget buys a guaranteed no-op."""
    y_pred = np.full(100, 0.5)
    attr = np.array(["A"] * 50 + ["B"] * 50)
    assert np.array_equal(y_pred, 1 - y_pred)

    result, messages = _caught(
        sensitivity_analysis,
        y_pred,
        attr,
        _dp_spread,
        perturbation_type="label_noise",
        perturbation_rate=0.10,
        n_iterations=50,
        random_state=1,
    )
    assert result.is_robust is None
    assert result.robustness_score is None
    assert any("own complement" in m for m in messages)


@pytest.mark.parametrize("ptype", ["label_noise", "group_noise", "subsample"])
def test_a_real_perturbation_still_returns_a_measured_robustness_verdict(ptype: str) -> None:
    """OVER-CORRECTION CONTROL, and the one that matters most: a guard that
    refused every perturbation would pass all four pins above."""
    rng = np.random.default_rng(0)
    attr = np.array(["A"] * 100 + ["B"] * 100)
    y_pred = np.concatenate([rng.binomial(1, 0.3, 100), rng.binomial(1, 0.7, 100)])

    result = sensitivity_analysis(
        y_pred,
        attr,
        _dp_spread,
        perturbation_type=ptype,
        perturbation_rate=0.10,
        n_iterations=50,
        random_state=1,
    )
    assert isinstance(result.is_robust, bool)
    assert result.robustness_score is not None
    assert math.isfinite(result.max_deviation)
    assert result.n_iterations_run == 50


# ---------------------------------------------------------------------------
# S2B02-04  the permutation design floor
#
# Two opposite errors in one expression. Unequal two-group splits were doubled
# for a label swap that is not an arrangement of their label multiset (a
# measurable shape refused); three or more groups dropped the relabeling
# multiplicity entirely (an unreachable shape advertised as detectable).
# ---------------------------------------------------------------------------


def test_an_unequal_two_group_split_is_no_longer_doubled() -> None:
    """OVER-CORRECTION PIN. 30 rows split 1/29 have exactly 30 arrangements and
    the most extreme data reaches 1/30; the old floor 2/30 = 0.0667 declared
    that 'no data could make this test significant' and refused."""
    y_pred = np.array([1] + [0] * 29)
    attr = np.array(["A"] + ["B"] * 29)
    # FIXTURE ASSERTION: an UNEQUAL two-group split, which is the branch at issue.
    counts = np.unique(attr, return_counts=True)[1]
    assert counts.size == 2 and counts[0] != counts[1]
    assert _label_arrangement_floor(attr, "two-sided") == pytest.approx(1 / 30)

    result = permutation_test_demographic_parity(y_pred, attr, n_permutations=10000, random_state=7)
    assert result.min_attainable_p_value == pytest.approx(1 / 30)
    assert result.detectable_at_05 is True
    assert math.isfinite(result.p_value)
    assert result.p_value < 0.05
    assert result.significant_at_05 is True
    assert result.method == "permutation"


def test_an_equal_two_group_split_keeps_its_doubling() -> None:
    """CONTROL for the same expression: the swap IS an arrangement when the two
    counts match, so 4 rows split 2/2 keep the 2/6 floor and the refusal."""
    attr = np.array(["F", "F", "M", "M"])
    assert _label_arrangement_floor(attr, "two-sided") == pytest.approx(2 / 6)

    result, messages = _caught(
        permutation_test_demographic_parity,
        np.array([1, 1, 0, 0]),
        attr,
        n_permutations=2000,
        random_state=0,
    )
    assert result.detectable_at_05 is False
    assert result.significant_at_05 is None
    assert math.isnan(result.p_value)
    assert messages


def test_three_groups_of_two_cannot_advertise_detectability() -> None:
    """UNDER-STATEMENT PIN. 6 rows in 3 groups of 2 have 90 arrangements and a
    two-sided statistic is unchanged by any of the 3! = 6 relabelings, so the
    floor is 6/90. Brute force over all 64 possible y_pred vectors at this shape
    puts the smallest reachable p at 0.2000; the old 1/90 = 0.0111 reported
    detectable_at_05=True and graded significant_at_05=False off it."""
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    attr = np.array(["A", "A", "B", "B", "C", "C"])
    counts = np.unique(attr, return_counts=True)[1]
    assert counts.size == 3 and set(counts) == {2}

    assert _label_arrangement_floor(attr, "two-sided") == pytest.approx(6 / 90)
    result, messages = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=10000, random_state=7
    )
    assert result.min_attainable_p_value == pytest.approx(6 / 90)
    assert result.detectable_at_05 is False
    assert result.significant_at_05 is None
    assert result.significant_at_01 is None
    assert math.isnan(result.p_value)
    assert messages


def test_the_refusal_names_the_cause_it_actually_had() -> None:
    """A design refused for having too few label arrangements was told 'the
    permutation null collapsed to 10000 usable draw(s) of 10000 requested' and
    handed method='permutation (not run: null distribution collapsed)'. The null
    did not collapse; the reader was sent after the wrong problem."""
    attr = np.array(["A", "A", "B", "B", "C", "C"])
    result, messages = _caught(
        permutation_test_demographic_parity,
        np.array([1, 1, 1, 0, 0, 0]),
        attr,
        n_permutations=10000,
        random_state=7,
    )
    assert result.method == "permutation (not run: too few label arrangements)"
    assert not any("null collapsed" in m for m in messages)
    assert any("too few distinct arrangements" in m for m in messages)


def test_a_genuinely_collapsed_null_still_says_so() -> None:
    """CONTROL for the message split: when the null really is the binding limit,
    the collapse sentence and its method string are what a reader gets."""

    def half_nan(y_pred: np.ndarray, attr: np.ndarray) -> float:
        # Measurable for the observed labels, unmeasurable for most shuffles.
        if attr[0] != "A":
            return float("nan")
        return _dp_spread(y_pred, attr)

    rng = np.random.default_rng(0)
    attr = np.array(["A"] * 50 + ["B"] * 50)
    y_pred = np.concatenate([rng.binomial(1, 0.2, 50), rng.binomial(1, 0.8, 50)])
    result, messages = _caught(
        permutation_test, y_pred, attr, half_nan, n_permutations=8, random_state=0
    )
    assert result.method == "permutation (not run: null distribution collapsed)"
    assert any("collapsed to" in m for m in messages)


def test_an_unknown_alternative_raises_instead_of_reporting_no_power() -> None:
    """The ValueError sat AFTER the design guard, so once that guard started
    refusing, a misconfiguration came back as a polite could-not-check."""
    with pytest.raises(ValueError, match="Unknown alternative"):
        permutation_test(
            np.array([1, 1, 0, 0]),
            np.array(["F", "F", "M", "M"]),  # a shape the design guard refuses
            _dp_spread,
            n_permutations=100,
            alternative="twosided",
        )


def test_equal_opportunity_floors_on_the_rows_its_statistic_reads() -> None:
    """The TPR spread reads only the y_true == 1 rows, but the floor was taken
    from the whole 200-row label array, so the design advertised the resample
    floor 1/10001 no matter how few positives there were."""
    attr = np.array(["F"] * 100 + ["M"] * 100)
    y_true = np.zeros(200, dtype=int)
    y_true[[0, 100]] = 1  # TWO positive-label rows in the whole dataset
    y_pred = np.zeros(200, dtype=int)
    y_pred[0] = 1
    assert int(np.sum(y_true == 1)) == 2

    result, messages = _caught(
        permutation_test_equal_opportunity,
        y_true,
        y_pred,
        attr,
        n_permutations=10000,
        random_state=1,
    )
    assert result.min_attainable_p_value is not None
    assert result.min_attainable_p_value > 0.05
    assert result.detectable_at_05 is False
    assert result.significant_at_05 is None
    assert math.isnan(result.p_value)
    assert result.method == "permutation (not run: effective design too small)"
    assert any("reads only 2 of 200 rows" in m for m in messages)


def test_an_equal_opportunity_design_with_real_power_still_measures() -> None:
    """OVER-CORRECTION CONTROL: the same wrapper on a design that CAN fire must
    still return a measured p-value, not a refusal."""
    rng = np.random.default_rng(0)
    attr = np.array(["F"] * 100 + ["M"] * 100)
    y_true = rng.binomial(1, 0.5, 200)
    y_pred = np.where(
        (y_true == 1) & (attr == "F"),
        rng.binomial(1, 0.3, 200),
        np.where(y_true == 1, rng.binomial(1, 0.95, 200), 0),
    )
    result = permutation_test_equal_opportunity(
        y_true, y_pred, attr, n_permutations=2000, random_state=1
    )
    assert math.isfinite(result.p_value)
    assert result.detectable_at_05 is True
    assert result.significant_at_05 is True
    assert result.method == "permutation"


# ---------------------------------------------------------------------------
# S2B02-05  a misconfigured loss reported as a could-not-check
#
# FairnessMetricType has seven members; FairnessAwareBCELoss implements five.
# The `raise ValueError("Unknown fairness metric")` is the LAST branch of the
# dispatch, and the single-group guard returns above it, so the misconfiguration
# came back as reason='single_group' plus a finite total loss.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from vfairness.in_processing.loss_functions.fairness_losses import (  # noqa: E402
    EqualizedOddsLoss,
    FairnessAwareBCELoss,
    create_fairness_loss,
)


def _batch(n: int = 24, *, one_group: bool = False, all_positive: bool = False):
    torch.manual_seed(0)
    y_pred = torch.rand(n)
    y_true = torch.ones(n) if all_positive else torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    if one_group:
        sensitive = torch.zeros(n)
    else:
        sensitive = torch.cat([torch.zeros(n // 4), torch.ones(n // 4)] * 2)
    return y_pred, y_true, sensitive


@pytest.mark.parametrize("metric", ["individual_fairness", "counterfactual_fairness"])
def test_a_metric_this_loss_cannot_compute_is_refused_at_construction(metric: str) -> None:
    """Before: FairnessAwareBCELoss(fairness_metric='individual_fairness') on a
    single-group batch returned 0.8980 with only the 'single_group' warning, and
    raised on the next batch that happened to have two groups."""
    with pytest.raises(ValueError, match="does not implement fairness_metric"):
        FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=0.5)


@pytest.mark.parametrize(
    "metric",
    [
        "demographic_parity",
        "equalized_odds",
        "equal_opportunity",
        "predictive_parity",
        "calibration",
    ],
)
def test_every_metric_this_loss_does_implement_is_still_accepted(metric: str) -> None:
    """OVER-CORRECTION CONTROL for the construction guard: the five implemented
    metrics must still build and still measure a healthy batch."""
    loss_fn = FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch()
    _total, components = loss_fn(y_pred, y_true, sensitive, return_components=True)
    assert components.batch_metrics["fairness_penalty_assessed"] is True
    assert math.isfinite(components.fairness_loss)


@pytest.mark.parametrize(
    "metric",
    [
        "demographic_parity",
        "equalized_odds",
        "equal_opportunity",
        "predictive_parity",
        "calibration",
    ],
)
def test_the_factory_refuses_a_single_group_batch_by_value_not_by_label(metric: str) -> None:
    """The substantive pin is NaN + assessed is False; the reason STRING is
    asserted separately so the pin does not rest on the wording."""
    loss_fn = create_fairness_loss(metric, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch(one_group=True)
    (_total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )
    assert math.isnan(components.fairness_loss)
    assert components.batch_metrics["fairness_penalty_assessed"] is False
    assert components.batch_metrics["fairness_unassessable_reason"] is not None
    # The finite value the optimizer actually saw is kept, not published.
    assert math.isfinite(components.batch_metrics["fairness_loss_unassessed_value"])
    assert messages


def test_the_bce_equalized_odds_arm_guard_refuses_a_half_defined_batch() -> None:
    """Coverage for FairnessAwareBCELoss._equalized_odds_penalty, which had
    none: disabling its refusal left 663 loss tests green."""
    loss_fn = FairnessAwareBCELoss(fairness_metric="equalized_odds", lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch(all_positive=True)
    assert bool((y_true == 0).sum() == 0), "the fixture must remove the FPR arm entirely"

    (_total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )
    assert math.isnan(components.fairness_loss)
    assert components.batch_metrics["fairness_penalty_assessed"] is False
    assert components.batch_metrics["fairness_unassessable_reason"] == "fpr_arm_not_comparable"
    # The arm that WAS measured stays visible instead of being passed off as
    # the whole criterion.
    assert components.batch_metrics["equalized_odds_arms_compared"] == ["tpr"]
    assert messages


def test_the_bce_equalized_odds_guard_refuses_a_single_group_batch() -> None:
    loss_fn = FairnessAwareBCELoss(fairness_metric="equalized_odds", lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch(one_group=True)
    (_total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )
    assert math.isnan(components.fairness_loss)
    assert components.batch_metrics["fairness_penalty_assessed"] is False
    assert components.batch_metrics["fairness_unassessable_reason"] == "single_group"
    assert messages


def test_the_bce_equalized_odds_penalty_still_measures_a_healthy_batch() -> None:
    """OVER-CORRECTION CONTROL for the two pins above."""
    loss_fn = FairnessAwareBCELoss(fairness_metric="equalized_odds", lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch()
    _total, components = loss_fn(y_pred, y_true, sensitive, return_components=True)
    assert components.batch_metrics["fairness_penalty_assessed"] is True
    assert components.batch_metrics["equalized_odds_arms_compared"] == ["fpr", "tpr"]
    assert math.isfinite(components.fairness_loss)


@pytest.mark.parametrize("tpr_weight,fpr_weight", [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_a_negative_arm_weight_is_refused_at_construction(
    tpr_weight: float, fpr_weight: float
) -> None:
    """`weights[arm] > 0` read a negative weight as 'not required' exactly like a
    zero. Measured: tpr_weight=-1, fpr_weight=1 returned fairness_loss=0.045965
    with fairness_penalty_assessed=True and no warning, on the FPR arm alone,
    published as equalized odds."""
    with pytest.raises(ValueError, match="must be finite and >= 0"):
        EqualizedOddsLoss(tpr_weight=tpr_weight, fpr_weight=fpr_weight, lambda_fairness=0.5)


def test_a_zero_arm_weight_is_still_a_legitimate_deliberate_exclusion() -> None:
    """OVER-CORRECTION CONTROL: 0.0 is how a caller excludes an arm on purpose,
    and it must still produce a measurement from the other one."""
    loss_fn = EqualizedOddsLoss(tpr_weight=0.0, fpr_weight=1.0, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch()
    _total, components = loss_fn(y_pred, y_true, sensitive, return_components=True)
    assert components.batch_metrics["fairness_penalty_assessed"] is True
    assert math.isfinite(components.fairness_loss)


def test_both_arm_weights_zero_is_still_the_could_not_check_it_was() -> None:
    loss_fn = EqualizedOddsLoss(tpr_weight=0.0, fpr_weight=0.0, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _batch()
    (_total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )
    assert math.isnan(components.fairness_loss)
    assert components.batch_metrics["fairness_unassessable_reason"] == "zero_arm_weights"
    assert messages
