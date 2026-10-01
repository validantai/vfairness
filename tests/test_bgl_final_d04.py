"""BGL final, group d04. Two capabilities, one rule: three states, never two.

  * ``identify_privileged_groups`` (evaluation/vfairness_metrics/intersectional.py)
  * ``permutation_test``          (evaluation/vfairness_metrics/robustness.py)

Both were recorded BGL-D with a reproduced defect. The reproduction was re-run at
the public entry on 2026-09-17 before anything was changed, and the results split
three ways:

ALREADY CORRECT (pinned here, because an unpinned correct behaviour is one
refactor from regressing):

  * intersectional: one cell with an unmeasurable FPR used to delete EVERY
    ``fpr_disparity`` finding in the run, measured critical ones included, because
    the skip test read ``if group_fpr is None`` and the engine uses NaN. The test
    is ``is_measured`` now and the unmeasurable cell is named in an explicit
    ``fpr_not_assessed`` entry instead of erasing the others.
  * intersectional: ``GroupAdvantage.to_dict`` serialises ``unmeasured``.
  * permutation_test: the two-sided doubling is the relabeling multiplicity, so an
    UNEQUAL two-group split is no longer doubled and no longer refused, and k>2
    groups no longer under-state the floor by the k! multiplicity.
  * permutation_test: the design-power refusal names its own cause and carries its
    own method string, instead of borrowing the collapsed-null sentence.
  * permutation_test: ``alternative`` is validated ABOVE the floor guard, so a
    misconfiguration cannot hide behind a could-not-check.

STILL REPRODUCED, FIXED HERE:

  * ``comprehensive_fairness_test`` ran every metric through ``permutation_test``
    with no ``n_design_rows``, although ``eo_diff`` reads only the rows with
    ``y_true == 1`` and ``pp_diff`` only the rows with ``y_pred == 1``. Measured at
    the public entry on 200 rows with 4 predicted and 4 true positives:
    ``permutation_test_equal_opportunity`` REFUSED the same statistic on the same
    data (min_attainable_p_value 0.060620, detectable_at_05 False,
    significant_at_05 None) while this function reported equal_opportunity
    min_attainable_p_value 0.000576 with detectable_at_05 True and a graded
    significant_at_05 False, and predictive_parity the same. Two graded negatives
    for tests no data at that shape could have made fire.

FIXED ELSEWHERE IN THIS REPAIR WAVE, PINNED HERE:

  * ``_norm_int_group`` in operations/pulse/orchestrator.py is the boundary
    between the engine (where NaN means could-not-check) and the JSON Pulse
    contract. It used to drop ``GroupAdvantage.unmeasured`` entirely and test the
    third state with ``value is None``, which is never true of the engine's NaN.
    Measured against the committed version on 2026-09-17: cell C came out as
    ``falsePositiveRate: NaN`` with no ``unmeasured`` key, both ratios came out
    NaN with ZERO warnings raised, and ``json.dumps(..., allow_nan=False)``
    refused the payload outright. The two Pulse tests at the end of this file
    are RED against that version and GREEN against the current tree.

Every refusal below is checked against a BRUTE FORCE over the actual arrangements,
not against the formula that produced it, and every block has a healthy-data
control that still measures.
"""

from __future__ import annotations

import itertools
import json
import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.intersectional import (
    GroupAdvantage,
    identify_privileged_groups,
    intersectional_disparity_analysis,
)
from vfairness.evaluation.vfairness_metrics.robustness import (
    comprehensive_fairness_test,
    permutation_test,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _caught(fn, *a, **kw):
    """Run and return (result, [UserWarning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _dp_diff(pred: np.ndarray, attr: np.ndarray) -> float:
    groups = np.unique(attr)
    if len(groups) < 2:
        return float("nan")
    rates = [float(np.mean(pred[attr == g])) for g in groups]
    return max(rates) - min(rates)


def _exact_two_sided_p(pred: np.ndarray, attr: np.ndarray) -> float:
    """The EXACT permutation p over every distinct label arrangement.

    Ground truth for the floor claims below. Not the formula under test: this
    enumerates, so a floor that disagrees with it is the floor's problem.
    """
    observed = abs(_dp_diff(pred, attr))
    seen: set = set()
    at_least_as_extreme = 0
    total = 0
    for perm in itertools.permutations(range(len(attr))):
        arrangement = tuple(attr[list(perm)])
        if arrangement in seen:
            continue
        seen.add(arrangement)
        total += 1
        if abs(_dp_diff(pred, np.asarray(arrangement))) >= observed - 1e-12:
            at_least_as_extreme += 1
    return at_least_as_extreme / total


def _healthy_split(
    n_per_group: int = 300, seed: int = 11
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A real, findable disparity on enough rows for every test to have power."""
    rng = np.random.default_rng(seed)
    y_pred = np.concatenate(
        [
            (rng.random(n_per_group) < 0.70).astype(int),
            (rng.random(n_per_group) < 0.30).astype(int),
        ]
    )
    y_true = np.concatenate(
        [
            (rng.random(n_per_group) < 0.55).astype(int),
            (rng.random(n_per_group) < 0.45).astype(int),
        ]
    )
    attr = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y_true, y_pred, attr


# ===========================================================================
# PART 1 - permutation_test
# ===========================================================================


# --- item 1: the two-sided doubling is only sound for EQUAL group sizes -----


def test_unequal_two_group_split_is_not_doubled_and_is_not_refused():
    """1/29 of 30 rows. Brute force says the design reaches 1/30 = 0.0333, so it
    CAN be significant; the old 2x doubling declared 0.0667 and refused it."""
    y_pred = np.array([1] + [0] * 29)
    attr = np.array(["A"] + ["B"] * 29)

    # Ground truth by enumeration, not by the formula under test. 30 distinct
    # arrangements; exactly one of them puts the single 'A' on the single 1.
    n_arrangements = math.factorial(30) // (math.factorial(1) * math.factorial(29))
    assert n_arrangements == 30
    extreme = sum(
        1
        for i in range(30)
        if abs(_dp_diff(y_pred, np.array(["B"] * i + ["A"] + ["B"] * (29 - i)))) >= 1.0 - 1e-12
    )
    assert extreme == 1
    assert extreme / n_arrangements == pytest.approx(1 / 30)

    result, _ = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=10000, random_state=0
    )
    assert result.min_attainable_p_value == pytest.approx(1 / 30)
    # and specifically NOT the doubled 2/30 the defect reported
    assert result.min_attainable_p_value != pytest.approx(2 / 30)
    assert result.detectable_at_05 is True
    assert result.method == "permutation"
    assert np.isfinite(result.p_value) and result.p_value < 0.05
    assert result.significant_at_05 is True


def test_equal_two_group_split_keeps_the_relabeling_doubling():
    """The over-correction control. When the two groups ARE the same size the
    swap is a real arrangement of this label multiset, so the doubling stands."""
    y_pred = np.array([1, 1, 0, 0])
    attr = np.array(["F", "F", "M", "M"])
    result, _ = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=2000, random_state=0
    )
    n_arrangements = math.factorial(4) // (math.factorial(2) * math.factorial(2))
    assert n_arrangements == 6
    assert result.min_attainable_p_value == pytest.approx(2 / 6)
    assert result.detectable_at_05 is False
    assert result.significant_at_05 is None


def test_a_one_sided_test_gets_no_doubling_at_all():
    """A one-sided reading is not invariant under the relabeling, so it keeps the
    conservative single arrangement."""
    y_pred = np.array([1, 1, 0, 0])
    attr = np.array(["F", "F", "M", "M"])
    result, _ = _caught(
        permutation_test,
        y_pred,
        attr,
        _dp_diff,
        n_permutations=500,
        alternative="greater",
        random_state=0,
    )
    assert result.min_attainable_p_value == pytest.approx(1 / 6)


# --- item 2: k > 2 groups must not under-state by the k! multiplicity ------


def test_three_groups_of_two_cannot_fire_and_says_so():
    """Brute force over every y_pred at this shape: the smallest reachable p is
    0.2. The defect advertised 1/90 = 0.0111 and detectable_at_05=True."""
    attr = np.array(["A", "A", "B", "B", "C", "C"])

    reachable = min(
        _exact_two_sided_p(np.array(bits), attr)
        for bits in itertools.product([0, 1], repeat=6)
        if len(set(bits)) > 1
    )
    assert reachable == pytest.approx(0.2), reachable
    assert reachable > 0.05  # no data at this shape can ever be significant

    y_pred = np.array([1, 1, 0, 0, 0, 0])
    result, msgs = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=10000, random_state=0
    )
    n_arrangements = math.factorial(6) // (math.factorial(2) ** 3)
    assert n_arrangements == 90
    # 6 = 3! relabelings of three same-size groups. A LOWER bound on 0.2, which
    # is all that is claimed, and already above alpha.
    assert result.min_attainable_p_value == pytest.approx(6 / 90)
    assert result.min_attainable_p_value != pytest.approx(1 / 90)
    assert result.detectable_at_05 is False
    assert np.isnan(result.p_value)
    assert result.significant_at_05 is None
    assert result.significant_at_01 is None
    assert any("too few distinct arrangements" in m for m in msgs), msgs


# --- item 3: the floor needs the EFFECTIVE design, not len(sensitive_attr) --


def _thin_design(n: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """200 rows; only 4 of them carry a true positive and only 4 a predicted
    positive, two per group. Both subset statistics are computable and both
    designs are far too thin for either to reach 0.05."""
    attr = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_pred = np.zeros(n, dtype=int)
    y_pred[[0, 1, n // 2, n // 2 + 1]] = 1
    y_true = np.zeros(n, dtype=int)
    y_true[[0, 2, n // 2, n // 2 + 2]] = 1
    return y_true, y_pred, attr


def test_equal_opportunity_wrapper_reports_the_subset_design_floor():
    y_true, y_pred, attr = _thin_design()
    result, msgs = _caught(
        permutation_test_equal_opportunity,
        y_true,
        y_pred,
        attr,
        n_permutations=10000,
        random_state=0,
    )
    # 4 of 200 rows are readable; the least likely intake of them has this
    # probability, so no p below it is reachable.
    assert result.min_attainable_p_value == pytest.approx(0.060620360686673605)
    assert result.min_attainable_p_value > 0.05
    assert result.detectable_at_05 is False
    assert np.isnan(result.p_value)
    assert result.significant_at_05 is None
    assert result.method == "permutation (not run: effective design too small)"
    assert any("reads only 4 of 200 rows" in m for m in msgs), msgs


def test_comprehensive_test_carries_the_effective_design_for_every_metric():
    """THE FIX. Same data, same statistic, same public library: the family had to
    stop grading two negatives the wrapper refuses."""
    y_true, y_pred, attr = _thin_design()
    wrapper, _ = _caught(
        permutation_test_equal_opportunity,
        y_true,
        y_pred,
        attr,
        n_permutations=2000,
        random_state=0,
    )
    results, _ = _caught(
        comprehensive_fairness_test, y_true, y_pred, attr, n_permutations=2000, random_state=0
    )
    tests = results["metric_tests"]

    eo = tests["equal_opportunity"]
    assert eo["min_attainable_p_value"] == pytest.approx(wrapper.min_attainable_p_value)
    assert eo["detectable_at_05"] is False
    assert eo["significant_at_05"] is None
    assert np.isnan(eo["p_value"])

    pp = tests["predictive_parity"]
    assert pp["min_attainable_p_value"] == pytest.approx(0.060620360686673605)
    assert pp["detectable_at_05"] is False
    assert pp["significant_at_05"] is None

    # demographic_parity reads every row, so its design is untouched and it is
    # still tested. The guard is per metric, not a blanket refusal.
    dp = tests["demographic_parity"]
    assert dp["detectable_at_05"] is True
    assert dp["significant_at_05"] in (True, False)

    assert sorted(results["not_assessable"]) == ["equal_opportunity", "predictive_parity"]
    assert results["n_tests"] == 1
    for name in ("equal_opportunity", "predictive_parity"):
        assert results["not_assessable_reasons"][name]


def test_the_thin_design_floor_is_a_lower_bound_not_an_invention():
    """The refusal above must be justified by the DESIGN, not by the formula.
    Enumerate every distinct assignment of the 200 labels to the 4 readable rows
    and confirm the reported floor is at or below the least likely one."""
    y_true, y_pred, attr = _thin_design()
    n, m = len(attr), int(np.sum(y_true == 1))
    assert m == 4
    counts = [int(c) for c in np.unique(attr, return_counts=True)[1]]
    # P(the 4 readable rows all come from one group) is the smallest intake
    # probability at this shape.
    least_likely = min(
        math.prod(math.comb(c, a) for c, a in zip(counts, intake)) / math.comb(n, m)
        for intake in itertools.product(range(m + 1), repeat=len(counts))
        if sum(intake) == m
    )
    reported = permutation_test_equal_opportunity(
        y_true, y_pred, attr, n_permutations=200, random_state=0
    ).min_attainable_p_value
    assert reported is not None
    assert reported <= least_likely + 1e-12
    assert reported > 0.05


# --- item 4: the refusal names its own cause ------------------------------


def test_the_enumeration_refusal_does_not_claim_the_null_collapsed():
    y_pred = np.array([1, 1, 0, 0, 0, 0])
    attr = np.array(["A", "A", "B", "B", "C", "C"])
    result, msgs = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=10000, random_state=0
    )
    assert result.method == "permutation (not run: too few label arrangements)"
    assert "collapsed" not in result.method
    joined = " ".join(msgs)
    assert "too few distinct arrangements" in joined
    assert "collapsed" not in joined, joined
    # The null did NOT collapse: every draw was usable.
    assert result.n_permutations == 10000


def test_a_genuinely_collapsed_null_still_says_collapsed():
    """The control for the sentence above. If the only refusal that can say
    'collapsed' is the one that never fires, the message split is decoration."""
    calls = {"n": 0}

    def flaky(pred, permuted):
        calls["n"] += 1
        # call 1 is the observed statistic; the next three draws are usable and
        # everything after that is unmeasurable.
        return 0.8 if calls["n"] <= 4 else float("nan")

    # >1000 rows, so the label enumeration floor is not computed and the
    # resample floor is the only one left to bind.
    attr = np.array(["A"] * 600 + ["B"] * 600)
    y_pred = np.zeros(1200, dtype=int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = permutation_test(y_pred, attr, flaky, n_permutations=100, random_state=0)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert result.method == "permutation (not run: null distribution collapsed)"
    assert result.n_permutations == 3
    assert result.min_attainable_p_value == pytest.approx(1 / 4)
    assert result.significant_at_05 is None
    assert any("permutation null collapsed to 3 usable draw(s) of 100" in m for m in msgs), msgs
    assert not any("too few distinct arrangements" in m for m in msgs), msgs


# --- item 5: a misconfiguration must not hide behind a refusal -------------


def test_an_unknown_alternative_raises_even_when_the_floor_guard_would_fire():
    """The shape matters. An unknown ``alternative`` is not "two-sided", so the
    enumeration floor is the UNDOUBLED 1/n_arrangements; the guard only fires
    underneath it on a shape where even that is above alpha. Four rows in a 2/2
    split give 1/6 = 0.167, so the guard DOES fire here, and before the
    validation moved above it a typo came back as a polite could-not-check about
    statistical power instead of the configuration error the caller made."""
    y_pred = np.array([1, 1, 0, 0])
    attr = np.array(["F", "F", "M", "M"])
    # the guard really does fire at this shape, with a spelling that IS valid
    guarded, _ = _caught(
        permutation_test,
        y_pred,
        attr,
        _dp_diff,
        n_permutations=50,
        alternative="greater",
        random_state=0,
    )
    assert guarded.min_attainable_p_value == pytest.approx(1 / 6)
    assert guarded.detectable_at_05 is False
    assert guarded.method.startswith("permutation (not run:")

    with pytest.raises(ValueError, match="Unknown alternative: twosided"):
        permutation_test(y_pred, attr, _dp_diff, n_permutations=50, alternative="twosided")


def test_a_valid_alternative_is_still_accepted():
    y_true, y_pred, attr = _healthy_split(n_per_group=120)
    for alternative in ("two-sided", "greater", "less"):
        result, _ = _caught(
            permutation_test,
            y_pred,
            attr,
            _dp_diff,
            n_permutations=400,
            alternative=alternative,
            random_state=1,
        )
        assert result.method == "permutation"


# --- CONTROL: healthy data still gets measured ----------------------------


def test_control_healthy_data_still_produces_a_real_permutation_test():
    y_true, y_pred, attr = _healthy_split()
    result, _ = _caught(
        permutation_test_demographic_parity, y_pred, attr, n_permutations=2000, random_state=3
    )
    assert result.method == "permutation"
    assert np.isfinite(result.p_value)
    assert result.p_value < 0.01
    assert result.significant_at_05 is True
    assert result.detectable_at_05 is True
    assert result.n_permutations == 2000


def test_control_comprehensive_test_on_healthy_data_tests_every_metric():
    """The over-correction guard for the fix above: adding n_design_rows must not
    start refusing designs that DO have power."""
    y_true, y_pred, attr = _healthy_split()
    results, _ = _caught(
        comprehensive_fairness_test, y_true, y_pred, attr, n_permutations=1000, random_state=3
    )
    assert results["not_assessable"] == []
    assert results["n_tests"] == 3
    assert results["any_significant"] is True
    assert "demographic_parity" in results["significant_metrics"]
    for name, test in results["metric_tests"].items():
        assert test["method"] == "permutation", name
        assert test["detectable_at_05"] is True, name
        assert np.isfinite(test["p_value"]), name


def test_control_a_moderately_thin_design_is_reported_but_still_run():
    """Between the two extremes: 6 readable rows give a floor of 0.0145, which is
    below alpha, so the test RUNS and simply carries the honest floor. The fix
    must report the subset design, not refuse every subset statistic."""
    n = 200
    attr = np.array(["A"] * 100 + ["B"] * 100)
    y_pred = np.zeros(n, dtype=int)
    y_pred[[0, 1, 2, 100, 101, 102]] = 1
    y_true = np.zeros(n, dtype=int)
    y_true[[0, 1, 3, 100, 101, 103]] = 1
    results, _ = _caught(
        comprehensive_fairness_test, y_true, y_pred, attr, n_permutations=1000, random_state=0
    )
    eo = results["metric_tests"]["equal_opportunity"]
    assert eo["min_attainable_p_value"] == pytest.approx(0.014465141011575979)
    assert eo["min_attainable_p_value"] < 0.05
    assert eo["detectable_at_05"] is True
    assert eo["method"] == "permutation"
    assert np.isfinite(eo["p_value"])
    assert results["not_assessable"] == []


# ===========================================================================
# PART 2 - identify_privileged_groups
# ===========================================================================


def _three_cells(include_unmeasurable: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A: 100 rows, all actually negative, FPR 0.90. B: 150 rows, all actually
    negative, FPR 0.033. C (optional): 50 rows whose labels are ALL positive, so
    its FPR denominator is empty and nobody can measure it."""
    y_true: List[int] = [0] * 100 + [0] * 150
    y_pred: List[int] = [1] * 90 + [0] * 10 + [1] * 5 + [0] * 145
    groups: List[str] = ["A"] * 100 + ["B"] * 150
    if include_unmeasurable:
        y_true += [1] * 50
        y_pred += [1] * 50
        groups += ["C"] * 50
    return np.array(y_true), np.array(y_pred), np.array(groups)


def _fpr_findings(analysis: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [f for f in analysis["findings"] if f["type"] == kind]


def test_one_unmeasurable_cell_does_not_erase_the_measured_fpr_findings():
    """The defect deleted EVERY fpr_disparity finding in the run, including two
    severity-critical ones, because one cell's NaN poisoned the pooled baseline."""
    y_true, y_pred, groups = _three_cells(include_unmeasurable=False)
    without, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    baseline = _fpr_findings(without, "fpr_disparity")
    assert len(baseline) == 2
    assert [f["severity"] for f in baseline] == ["critical", "critical"]

    y_true, y_pred, groups = _three_cells(include_unmeasurable=True)
    with_c, msgs = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    kept = _fpr_findings(with_c, "fpr_disparity")
    assert len(kept) == 2, "an unmeasurable cell deleted the measured findings"
    assert {f["groups"][0] for f in kept} == {"A", "B"}
    assert [f["severity"] for f in kept] == ["critical", "critical"]
    for finding in kept:
        assert np.isfinite(finding["metric_values"]["overall_fpr"])
        assert np.isfinite(finding["metric_values"]["fpr_delta"])

    # and the cell nobody could measure is NAMED, so "no finding for C" can never
    # be read as "C was checked and was fine".
    not_assessed = _fpr_findings(with_c, "fpr_not_assessed")
    assert len(not_assessed) == 1
    assert not_assessed[0]["groups"] == ["C"]
    assert not_assessed[0]["p_value"] is None
    assert not_assessed[0]["fpr_comparison_ran"] is True
    assert "could not be measured" in not_assessed[0]["description"]
    assert any("false positive rate could not be" in m for m in msgs), msgs


def test_the_unmeasurable_cell_is_kept_out_of_the_pooled_baseline():
    """Not just 'findings survived': the baseline must be the one computed from
    the measured cells alone, so the surviving numbers are the right numbers."""
    y_true, y_pred, groups = _three_cells(include_unmeasurable=True)
    analysis, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    pooled = {f["metric_values"]["overall_fpr"] for f in _fpr_findings(analysis, "fpr_disparity")}
    assert len(pooled) == 1
    # 90 false positives out of 250 actual negatives across A and B only.
    assert pooled.pop() == pytest.approx(95 / 250, abs=5e-4)


def test_group_advantage_to_dict_serialises_the_unmeasured_map():
    """The pin the auditor asked for by name: deleting this key from to_dict left
    every other test in the suite green."""
    y_true, y_pred, groups = _three_cells(include_unmeasurable=True)
    analysis, _ = _caught(identify_privileged_groups, y_true, y_pred, groups, min_group_size=30)
    rows = {g.group: g.to_dict() for g in analysis["all_groups"]}

    assert "unmeasured" in rows["C"], "to_dict dropped the reason the value is missing"
    assert "false_positive_rate" in rows["C"]["unmeasured"]
    assert "empty" in rows["C"]["unmeasured"]["false_positive_rate"]
    assert np.isnan(rows["C"]["false_positive_rate"])

    # CONTROL: a measured cell carries the key holding an EMPTY map, so a reader
    # can tell "everything here is real" from "this producer does not report".
    for name in ("A", "B"):
        assert rows[name]["unmeasured"] == {}
        assert np.isfinite(rows[name]["false_positive_rate"])

    # and the map survives a round trip through the serialisation it exists for
    assert json.loads(json.dumps(rows["C"]["unmeasured"]))["false_positive_rate"]


def test_to_dict_carries_a_reason_for_every_field_it_reports_as_nan():
    """The rule stated as a rule, so a NEW unmeasurable field cannot be added
    without its reason: no NaN in the row may be unexplained."""
    y_true = np.array([0] * 60 + [1] * 60)
    y_pred = np.zeros(120, dtype=int)
    groups = np.array(["X"] * 60 + ["Y"] * 60)
    analysis, msgs = _caught(identify_privileged_groups, y_true, y_pred, groups, min_group_size=30)
    assert any("could not be measured" in m for m in msgs), msgs
    for cell in analysis["all_groups"]:
        row = cell.to_dict()
        for field, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                assert field in row["unmeasured"], f"{cell.group}.{field} is NaN with no reason"
                assert row["unmeasured"][field]


def test_an_unmeasured_map_is_never_shared_between_cells():
    """``field(default_factory=dict)`` is load bearing. A shared default would
    make one cell's reason appear on every other cell."""
    first = GroupAdvantage(
        group="a",
        positive_rate=0.5,
        size=10,
        relative_to_overall=1.0,
        relative_to_best=1.0,
        disparity_contribution=0.0,
        severity="info",
    )
    second = GroupAdvantage(
        group="b",
        positive_rate=0.5,
        size=10,
        relative_to_overall=1.0,
        relative_to_best=1.0,
        disparity_contribution=0.0,
        severity="info",
    )
    first.unmeasured["false_positive_rate"] = "no actual negatives"
    assert second.unmeasured == {}
    assert second.to_dict()["unmeasured"] == {}
    # to_dict must copy, or a caller mutating the returned dict edits the cell
    snapshot = first.to_dict()["unmeasured"]
    snapshot["injected"] = "x"
    assert "injected" not in first.unmeasured


# --- CONTROL: healthy data still gets measured ----------------------------


def test_control_healthy_data_reports_no_unmeasured_field_anywhere():
    y_true, y_pred, groups = _healthy_split()
    analysis, msgs = _caught(identify_privileged_groups, y_true, y_pred, groups, min_group_size=30)
    assert analysis["privileged_group"] is not None
    assert analysis["disparity_severity"] != "not_assessed"
    assert np.isfinite(analysis["max_disparity"])
    assert analysis["max_disparity"] > 0.2
    for cell in analysis["all_groups"]:
        assert cell.unmeasured == {}
        assert cell.to_dict()["unmeasured"] == {}
        assert np.isfinite(cell.false_positive_rate)
        assert np.isfinite(cell.relative_to_best)
        assert np.isfinite(cell.relative_to_overall)
    assert not any("could not be measured" in m for m in msgs), msgs


def test_control_healthy_data_still_raises_fpr_disparity_findings():
    y_true, y_pred, groups = _healthy_split()
    analysis, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    assert _fpr_findings(analysis, "fpr_not_assessed") == []
    assert len(_fpr_findings(analysis, "fpr_disparity")) >= 1


def test_control_a_genuinely_zero_fpr_is_still_a_measurement():
    """The reverse direction. A cell with real negatives and no false positives
    has a MEASURED FPR of 0.0, and refusing it would throw away evidence."""
    y_true = np.array([0] * 80 + [0] * 80)
    y_pred = np.array([0] * 80 + [1] * 60 + [0] * 20)
    groups = np.array(["clean"] * 80 + ["flagged"] * 80)
    analysis, _ = _caught(identify_privileged_groups, y_true, y_pred, groups, min_group_size=30)
    rows = {g.group: g for g in analysis["all_groups"]}
    assert rows["clean"].false_positive_rate == 0.0
    assert not math.isnan(rows["clean"].false_positive_rate)
    assert rows["clean"].unmeasured == {}
    assert rows["flagged"].false_positive_rate == pytest.approx(0.75)


# --- the Pulse boundary: the reason has to reach the consumer -------------


def test_the_pulse_normaliser_carries_the_reason_and_emits_no_bare_nan():
    """``_norm_int_group`` is the boundary between the engine (NaN means
    could-not-check) and the JSON Pulse contract. A NaN that crosses it reaches
    ``json.dump(..., allow_nan=True)`` in the task handler and is written as the
    bare ``NaN`` token, which the browser's JSON.parse rejects outright."""
    from vfairness.operations.pulse.orchestrator import _normalise_intersectional

    y_true, y_pred, groups = _three_cells(include_unmeasurable=True)
    analysis, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    payload, _ = _caught(_normalise_intersectional, analysis, False)
    cells = {c["group"]: c for c in payload["allGroups"]}

    assert cells["C"]["falsePositiveRate"] is None
    assert "unmeasured" in cells["C"], "the reason was dropped at the Pulse boundary"
    reasons = cells["C"]["unmeasured"]
    assert any("falsePositiveRate" == k or "false_positive_rate" == k for k in reasons)
    assert any("empty" in str(v) for v in reasons.values())

    for name in ("A", "B"):
        assert cells[name]["unmeasured"] == {}
        assert cells[name]["falsePositiveRate"] is not None

    # strict JSON, which is what the transport and the browser actually do
    json.dumps(payload["allGroups"], allow_nan=False)


def test_the_pulse_normaliser_warns_when_a_ratio_could_not_be_measured():
    """NaN, not None, is how the engine says could-not-check for the two ratios.
    A three-state test written as ``value is None`` never fires on it."""
    from vfairness.operations.pulse.orchestrator import _normalise_intersectional

    y_true = np.array([0] * 60 + [1] * 60)
    y_pred = np.zeros(120, dtype=int)
    groups = np.array(["X"] * 60 + ["Y"] * 60)
    analysis, _ = _caught(
        intersectional_disparity_analysis, y_true, y_pred, groups, min_group_size=30
    )
    payload, msgs = _caught(_normalise_intersectional, analysis, False)
    for cell in payload["allGroups"]:
        assert cell["relativeToOverall"] is None
        assert cell["relativeToBest"] is None
        assert cell["unmeasured"], cell
    assert any("relativeToOverall" in m for m in msgs), msgs
    json.dumps(payload["allGroups"], allow_nan=False)
