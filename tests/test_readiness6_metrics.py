"""READINESS-6: the evaluation metrics that graded something nobody measured.

Nine defects, each reproduced by execution on 2026-09-10 before it was fixed,
each pinned here by a REFUSAL test and an OVER-CORRECTION CONTROL. The controls
are not decoration: every one of these fixes removes a verdict, and a fix that
removes the REAL verdicts too would pass the refusal half on its own.

The shared design-power helpers (`min_attainable_p_permutation`, `detectability`)
are pinned separately in ``tests/test_readiness6_design_power.py``; this file
pins the nine call sites that now use them or that had the same shape of hole.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)
from vfairness.evaluation.vfairness_metrics._statistics import (
    apply_multiple_testing_correction,
    benjamini_hochberg_correction,
    compute_metric_with_ci,
)
from vfairness.evaluation.vfairness_metrics.fairness_decomposition import (
    _grade,
    fairness_decomposition,
)
from vfairness.evaluation.vfairness_metrics.intersectional import (
    GroupAdvantage,
    generate_structured_findings,
)
from vfairness.evaluation.vfairness_metrics.recourse import (
    _column_sampler,
    generate_recourse,
)
from vfairness.evaluation.vfairness_metrics.report import (
    _metric_proportion_p_values,
    _threshold_entry,
    classification_fairness_report,
)
from vfairness.evaluation.vfairness_metrics.robustness import (
    comprehensive_fairness_test,
    permutation_test,
)
from vfairness.explainer import _explain_experiment_result
from vfairness.operations.experimentation.experiment import (
    DesignType,
    ExperimentResult,
)

# Labels are dtype=object throughout: a numpy '<U5' array silently TRUNCATES
# longer group names, which turns two distinct strata into one.


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ===========================================================================
# 1. robustness.permutation_test: a collapsed null produced a finite phantom
# ===========================================================================


def _null_after(observed, usable, filler=0.01):
    """A metric_func: a finite observed statistic, then `usable` measurable
    permutations and NaN for the rest."""
    state = {"i": 0}

    def metric_func(y_pred, attr):
        state["i"] += 1
        if state["i"] == 1:
            return observed
        return filler if state["i"] <= usable + 1 else float("nan")

    return metric_func


def _pred_and_attr(n=200):
    attr = np.array(["A"] * (n // 2) + ["B"] * (n // 2), dtype=object)
    y_pred = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    return y_pred, attr


class TestACollapsedPermutationNullIsCouldNotCheck:
    def test_three_usable_draws_is_refused_not_graded(self):
        """REFUSAL PIN. Measured before the fix: p_value=1.0,
        n_permutations=3, significant_at_05=False, for a test whose p-floor is
        0.25 and which therefore could never have been significant."""
        y_pred, attr = _pred_and_attr()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = permutation_test(
                y_pred, attr, _null_after(0.62, 3), n_permutations=100, random_state=1
            )

        assert math.isnan(result.p_value), f"got a finite phantom p={result.p_value!r}"
        assert result.significant_at_05 is None
        assert result.significant_at_01 is None
        assert result.n_permutations == 3
        assert result.min_attainable_p_value == pytest.approx(0.25)
        assert result.detectable_at_05 is False
        assert "NOT DETECTABLE" in result.design_note
        assert any("COULD NOT CHECK" in str(w.message) for w in caught)

    def test_an_emptied_null_does_not_certify_an_eighty_point_gap(self):
        """REFUSAL PIN. Measured before the fix: an 80-point demographic-parity
        gap with EVERY permutation unmeasurable came back p_value=1.0,
        significant_at_05=False. Certified not significant at p=1.000."""
        y_pred, attr = _pred_and_attr()
        result = _quiet(
            permutation_test,
            y_pred,
            attr,
            _null_after(0.80, 0),
            n_permutations=500,
            random_state=1,
        )

        assert result.n_permutations == 0
        assert math.isnan(result.p_value)
        assert result.significant_at_05 is None
        assert result.min_attainable_p_value == 1.0

    def test_the_point_zero_one_verdict_has_its_own_floor(self):
        """50 usable draws floor at 0.0196: enough to answer at 0.05, never
        enough to answer at 0.01. Two states there would be one state too few."""
        y_pred, attr = _pred_and_attr()
        result = _quiet(
            permutation_test,
            y_pred,
            attr,
            _null_after(0.90, 50),
            n_permutations=100,
            random_state=1,
        )

        assert result.significant_at_05 is not None
        assert result.significant_at_01 is None, (
            "a design that cannot reach 0.01 must not report a measured negative there"
        )

    def test_a_real_permutation_test_still_reaches_significance(self):
        """OVER-CORRECTION CONTROL, and the one that matters most: a fix that
        refused every permutation test would pass all three tests above."""
        y_pred, attr = _pred_and_attr()

        def dp(pred, a):
            groups = np.unique(a)
            rates = [float(np.mean(pred[a == g])) for g in groups]
            return max(rates) - min(rates)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = permutation_test(y_pred, attr, dp, n_permutations=1000, random_state=1)

        assert math.isfinite(result.p_value)
        assert bool(result.significant_at_05) is True
        assert bool(result.significant_at_01) is True
        assert result.design_note == ""
        assert result.detectable_at_05 is True
        assert not caught


class TestOnePhantomBuriedTheOneRealFinding:
    """The family effect, which is what actually reached Pulse.

    `comprehensive_fairness_test` is the sole `statistical` input to Pulse. On
    this fixture, measured 2026-09-10 before the fix:

        demographic_parity  p=0.0495  n_permutations=100  sig05=True
        equal_opportunity   p=1.0     n_permutations=3    sig05=False
        predictive_parity   p=nan     n_permutations=0    sig05=None

    Family entering Benjamini-Hochberg [0.0495, 1.0], m=2, adjusted 0.0990:
    significant_metrics=[] and any_significant=False. Two unmeasurable
    comparisons, two different treatments, and the one real finding buried.
    """

    @staticmethod
    def _fixture():
        n, nb = 200, 6
        na = n - nb
        attr = np.array(["A"] * na + ["B"] * nb, dtype=object)
        y_pred = np.zeros(n, dtype=int)
        y_pred[: int(round(na * 0.35))] = 1
        y_true = np.zeros(n, dtype=int)
        y_true[0] = 1
        y_true[n - 1] = 1
        return y_true, y_pred, attr

    def test_the_real_finding_survives_now(self):
        y_true, y_pred, attr = self._fixture()
        res = _quiet(
            comprehensive_fairness_test,
            y_true,
            y_pred,
            attr,
            n_permutations=100,
            random_state=2,
        )

        assert res["n_tests"] == 1, f"family is {res['p_values']}"
        assert res["significant_metrics"] == ["demographic_parity"]
        assert res["any_significant"] is True
        assert set(res["not_assessable"]) == {"equal_opportunity", "predictive_parity"}
        # The reason travels with the refusal, so a reader is not left to guess
        # which of the two kinds of could-not-check happened.
        assert "NOT DETECTABLE" in res["not_assessable_reasons"]["equal_opportunity"]

    def test_the_honest_family_is_the_one_that_ran(self):
        """Arithmetic control: m=2 does not reject, m=1 does. If the phantom
        ever gets back in, this is the number that changes."""
        y_true, y_pred, attr = self._fixture()
        res = _quiet(
            comprehensive_fairness_test,
            y_true,
            y_pred,
            attr,
            n_permutations=100,
            random_state=2,
        )
        real_p = res["p_values"][0]
        assert 0.025 < real_p < 0.05, "fixture no longer sits in the band that flips"
        with_phantom = _quiet(benjamini_hochberg_correction, np.array([real_p, 1.0]))
        assert not bool(with_phantom.rejection_mask[0])
        honest = _quiet(benjamini_hochberg_correction, np.array([real_p]))
        assert bool(honest.rejection_mask[0])

    def test_a_healthy_run_is_untouched(self):
        """OVER-CORRECTION CONTROL: three measurable metrics, all still tested."""
        rng = np.random.default_rng(4)
        n = 600
        attr = np.array(["A"] * 300 + ["B"] * 300, dtype=object)
        y_true = rng.binomial(1, 0.5, n)
        y_pred = np.where(attr == "A", rng.binomial(1, 0.8, n), rng.binomial(1, 0.2, n))
        res = _quiet(
            comprehensive_fairness_test,
            y_true,
            y_pred,
            attr,
            n_permutations=500,
            random_state=5,
        )

        assert res["n_tests"] == 3
        assert res["not_assessable"] == []
        assert "demographic_parity" in res["significant_metrics"]
        assert res["undetectable_metrics"] == {}


# ===========================================================================
# 2. intersectional: a finding that makes no claim was given a neutral p
# ===========================================================================


def _group(name, size, pred, gt, fpr=0.20):
    return GroupAdvantage(
        group=name,
        positive_rate=pred,
        size=size,
        relative_to_overall=1.0,
        relative_to_best=1.0,
        disparity_contribution=0.0,
        severity="info",
        ground_truth_rate=gt,
        false_positive_rate=fpr,
        prediction_delta=pred - gt,
    )


def _mitigating_analysis():
    """Five real findings plus one informational 'accurate_reflection'.

    The flagged cell is a 7-point over-prediction on 660 people, p=0.008968.
    Measured 2026-09-10: at m=5 (the tests that RAN) it corrects to 0.044840 and
    is SIGNIFICANT; at m=6 (with the informational entry counted as a
    hypothesis) it corrects to 0.053808 and is not.
    """
    groups = [
        _group("female|black", 660, 0.42, 0.35),
        _group("male|white", 60, 0.62, 0.55),
        _group("female|white", 55, 0.40, 0.47),
        _group("male|black", 50, 0.36, 0.29),
        _group("nb|asian", 52, 0.44, 0.37),
    ]
    return {
        "all_groups": groups,
        "max_disparity": 0.23,
        "max_ground_truth_disparity": 0.30,  # amplification -0.07 -> mitigation
        "privileged_group": groups[1],
        "disadvantaged_group": groups[3],
    }


class TestAnInformationalFindingIsNotAHypothesis:
    def test_the_seven_point_over_prediction_is_significant_again(self):
        """REFUSAL PIN, from the reader's side: the verdict, not the plumbing."""
        findings = _quiet(generate_structured_findings, _mitigating_analysis())
        over = [f for f in findings if f["type"] == "over_prediction"]
        flagged = min(over, key=lambda f: f["p_value"])

        assert flagged["p_value"] == pytest.approx(0.008968, rel=1e-3)
        assert flagged["p_value_corrected"] == pytest.approx(0.04484, rel=1e-3)
        assert flagged["statistically_significant"] is True

    def test_the_informational_finding_carries_no_verdict_at_all(self):
        """Three states. It is not significant, it is not NOT-significant, and
        it is not a family member: no hypothesis was tested."""
        findings = _quiet(generate_structured_findings, _mitigating_analysis())
        info = [f for f in findings if f["type"] == "accurate_reflection"]
        assert len(info) == 1, "fixture no longer produces the informational finding"

        assert info[0]["p_value"] is None
        assert info[0]["p_value_corrected"] is None
        assert info[0]["statistically_significant"] is None
        assert "not part of the multiple-comparison family" in info[0]["not_tested_reason"]

    def test_the_family_is_five_not_six(self):
        findings = _quiet(generate_structured_findings, _mitigating_analysis())
        tested = [f for f in findings if f["p_value"] is not None]
        assert len(findings) == 6
        assert len(tested) == 5

    def test_a_genuinely_weak_finding_is_still_refused(self):
        """OVER-CORRECTION CONTROL. Dropping the phantom must not turn every
        remaining finding significant: the four noisy cells stay False (measured
        and not significant), which is a DIFFERENT answer from None."""
        findings = _quiet(generate_structured_findings, _mitigating_analysis())
        weak = [f for f in findings if f.get("p_value") is not None and f["p_value"] > 0.4]
        assert len(weak) == 4
        for f in weak:
            assert f["statistically_significant"] is False


# ===========================================================================
# 3. report: the same hypothesis corrected up to three times
# ===========================================================================


def _duplicate_hypothesis_fixture():
    """1000 rows, seed 85. Measured before the fix, the family entering the
    correction was [0.016225, 0.016225, 0.037311, 0.604939, 0.037311]: m=5 for
    3 distinct hypotheses. Bonferroni at m=5 rejects nothing (0.0811); at the
    honest m=3 the real demographic-parity finding IS rejected (0.048675)."""
    rng = np.random.default_rng(85)
    n = 1000
    attr = np.array(["A"] * 500 + ["B"] * 500, dtype=object)
    y_true = rng.binomial(1, 0.45, n)
    y_pred = np.where(attr == "A", rng.binomial(1, 0.50, n), rng.binomial(1, 0.47, n))
    return y_true, y_pred, attr


class TestOneFamilyMemberPerDistinctHypothesis:
    def test_the_duplicates_are_folded_and_disclosed(self):
        y_true, y_pred, attr = _duplicate_hypothesis_fixture()
        from vfairness.evaluation.vfairness_metrics._grouping import GroupManager

        names, p_values, shared = _quiet(
            _metric_proportion_p_values, y_true, y_pred, GroupManager(attr, min_group_size=30)
        )

        assert names == [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "predictive_parity_difference",
        ]
        assert len(p_values) == 3
        assert shared == {
            "demographic_parity_difference": ["demographic_parity_ratio"],
            "equal_opportunity_difference": ["equalized_odds_difference"],
        }

    def test_the_buried_finding_is_reported_again(self):
        """REFUSAL PIN, from the reader's side."""
        y_true, y_pred, attr = _duplicate_hypothesis_fixture()
        corr = _quiet(
            classification_fairness_report,
            y_true,
            y_pred,
            attr,
            include_ci=True,
            n_bootstrap=200,
            multiple_testing_correction="bonferroni",
            random_state=0,
        )["multiple_testing_correction"]

        assert corr["tested_metrics"][0] == "demographic_parity_difference"
        assert corr["original_p_values"][0] == pytest.approx(0.016225, rel=1e-3)
        assert corr["adjusted_p_values"][0] == pytest.approx(0.048675, rel=1e-3)
        assert bool(corr["rejection_mask"][0]) is True
        assert corr["n_rejected"] == 1

    def test_a_blatant_gap_is_still_significant(self):
        """OVER-CORRECTION CONTROL: shrinking the family must not be achieved by
        dropping metrics that test something of their own. Same shape as
        test_audit_wave1's pin, which indexes tested_metrics by name."""
        rng = np.random.default_rng(3)
        n = 2000
        attr = np.array(["A"] * 1000 + ["B"] * 1000, dtype=object)
        y_true = rng.binomial(1, 0.5, n)
        y_pred = np.where(attr == "A", rng.binomial(1, 0.8, n), rng.binomial(1, 0.2, n))
        corr = _quiet(
            classification_fairness_report,
            y_true,
            y_pred,
            attr,
            include_ci=True,
            n_bootstrap=50,
            multiple_testing_correction="bonferroni",
        )["multiple_testing_correction"]

        names = corr["tested_metrics"]
        assert len(names) == len(corr["adjusted_p_values"]), "names must stay aligned"
        idx = names.index("demographic_parity_difference")
        assert bool(np.asarray(corr["rejection_mask"])[idx])
        # The folded metrics are still named, so nothing disappeared silently.
        assert "demographic_parity_ratio" in corr["metrics_sharing_a_tested_hypothesis"].get(
            "demographic_parity_difference", []
        )


# ===========================================================================
# 4. report._threshold_entry: a bound that cannot be breached grades nothing
# ===========================================================================


class TestTheReportAgreesWithTheSharedThresholdRule:
    @pytest.mark.parametrize(
        "metric_name,value,threshold",
        [
            # A required MINIMUM of 0.0 is met by every value, the worst
            # included. Measured before the fix: PASS.
            ("demographic_parity_ratio", 0.00, 0.0),
            ("disparate_impact_ratio", 0.00, 0.0),
            # The OVER-CORRECTION that was already here: a NEGATIVE maximum
            # returned FAIL where the shared rule returns COULD_NOT_CHECK.
            ("demographic_parity_difference", 0.42, -0.1),
            ("calibration_difference", 0.01, -0.5),
        ],
    )
    def test_a_degenerate_bound_is_not_assessable(self, metric_name, value, threshold):
        entry = _threshold_entry(metric_name, value, threshold)
        outcome, message = check_threshold(metric_name, value, threshold)

        assert outcome is ThresholdOutcome.COULD_NOT_CHECK
        assert entry["status"] == "NOT_ASSESSABLE", (
            f"_threshold_entry says {entry['status']} where the shared rule says "
            f"{outcome.name} on identical input"
        )
        assert entry["threshold"] is None
        assert entry["reason"] == message, "the reason must come from the shared rule"

    def test_a_never_selected_group_no_longer_lifts_the_fairness_score(self):
        """REFUSAL PIN. Measured before the fix: a model that NEVER selects
        group B, graded against a degenerate 0.0 minimum, put
        demographic_parity_ratio into passed_metrics and lifted fairness_score
        from 0.00 to 0.25."""
        rng = np.random.default_rng(11)
        n = 1200
        attr = np.array(["A"] * 600 + ["B"] * 600, dtype=object)
        y_pred = np.where(attr == "A", 1, 0)
        y_true = rng.binomial(1, 0.5, n)

        rep = _quiet(
            classification_fairness_report,
            y_true,
            y_pred,
            attr,
            thresholds={"demographic_parity_ratio": 0.0},
        )
        assessment = rep["assessment"]

        assert assessment["fairness_score"] == 0.0
        assert [m["metric"] for m in assessment["passed_metrics"]] == []
        not_assessable = {m["metric"] for m in assessment["not_assessable_metrics"]}
        assert "demographic_parity_ratio" in not_assessable

    @pytest.mark.parametrize(
        "metric_name,value,threshold,expected",
        [
            # OVER-CORRECTION CONTROLS: real bounds still grade, in both
            # directions, including the float-representation boundary that
            # `_within` exists for and that a plain `<` gets wrong.
            ("disparate_impact_ratio", 0.80, 0.80, "PASS"),
            ("disparate_impact_ratio", 0.7999, 0.80, "FAIL"),
            ("disparate_impact_ratio", 1.00, 0.80, "PASS"),
            ("disparate_impact_ratio", 0.00, 0.80, "FAIL"),
            ("demographic_parity_difference", 0.10, 0.10, "PASS"),
            ("demographic_parity_difference", 1.00, 0.10, "FAIL"),
            # A maximum of 0.0 on a violation magnitude is a real zero-tolerance
            # policy and MUST keep grading; only the negative mirror is
            # degenerate.
            ("demographic_parity_difference", 0.00, 0.0, "PASS"),
            ("demographic_parity_difference", 0.01, 0.0, "FAIL"),
        ],
    )
    def test_real_bounds_still_grade(self, metric_name, value, threshold, expected):
        assert _threshold_entry(metric_name, value, threshold)["status"] == expected

    def test_the_float_representation_tolerance_survives_the_shared_rule(self):
        """OVER-CORRECTION CONTROL for the fix ITSELF.

        Routing the degenerate-bound question through `check_threshold` must not
        drag the plain `<` comparison along with it. `_within` exists because a
        metric sitting EXACTLY on its threshold arrives with representation
        noise (0.80 - 0.70 is 0.10000000000000009, 0.1 + 0.7 is
        0.7999999999999999), and without the tolerance the verdict depends on
        binary representation rather than on fairness. Caught by sabotage: a
        version that graded with `float(value) >= float(threshold)` passed every
        other case in this class.
        """
        noisy_gap = 0.80 - 0.70  # 0.10000000000000009
        assert noisy_gap != 0.10, "fixture no longer carries representation noise"
        assert (
            _threshold_entry("demographic_parity_difference", noisy_gap, 0.10)["status"] == "PASS"
        )

        noisy_ratio = 0.1 + 0.7  # 0.7999999999999999
        assert noisy_ratio != 0.80, "fixture no longer carries representation noise"
        assert _threshold_entry("disparate_impact_ratio", noisy_ratio, 0.80)["status"] == "PASS"

    def test_a_well_powered_report_still_produces_its_real_score(self):
        """OVER-CORRECTION CONTROL. A fix that answered NOT_ASSESSABLE more
        readily would leave fairness_score None on a perfectly ordinary run."""
        rng = np.random.default_rng(19)
        n = 2000
        attr = np.array(["A"] * 1000 + ["B"] * 1000, dtype=object)
        y_true = rng.binomial(1, 0.5, n)
        y_pred = rng.binomial(1, 0.5, n)  # no disparity: a fair model

        assessment = _quiet(classification_fairness_report, y_true, y_pred, attr)["assessment"]

        assert assessment["fairness_score"] == 1.0
        assert len(assessment["passed_metrics"]) >= 4
        assert assessment["failed_metrics"] == []


# ===========================================================================
# 5 + 6. fairness_decomposition
# ===========================================================================


def _head_biased_dataset(n=2000):
    """The first 200 rows carry NO disparity; the remaining 1800 carry all of
    it. Measured before the fix, the default max_rows=200 HEAD SLICE reported
    severity 'low' at +0.0002 where the data supplied gives 'critical' at
    +0.7200, with 1800 of 2000 rows discarded and nothing said about it."""
    rng = np.random.default_rng(3)
    prot = np.array((["A", "B"] * (n // 2)), dtype=object)
    proxy = np.where(prot == "A", 1.0, 0.0)
    head = min(200, n)
    proxy[:head] = rng.normal(0, 0.006, head)
    X = np.column_stack([rng.normal(0, 1, n), proxy, rng.normal(0, 1, n)])
    return X, prot


def _predict_from_proxy(Xa):
    return np.clip(0.10 + 0.80 * np.asarray(Xa, dtype=float)[:, 1], 0.0, 1.0)


class TestTheDecompositionMeasuresTheDataItWasGiven:
    def test_the_budget_is_a_sample_not_the_first_rows(self):
        """REFUSAL PIN: the verdict must not depend on row order."""
        X, prot = _head_biased_dataset()
        res = _quiet(
            fairness_decomposition,
            _predict_from_proxy,
            X,
            prot,
            feature_names=["f0", "proxy", "f2"],
            n_permutations=8,
            random_state=1,
        )

        assert res.severity == "critical", f"got {res.severity} at {res.total_disparity:+.4f}"
        assert res.total_disparity == pytest.approx(0.72, abs=0.05)
        assert "proxy" in res.flagged_proxies

    def test_the_truncation_is_disclosed_three_ways(self):
        X, prot = _head_biased_dataset()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fairness_decomposition(
                _predict_from_proxy,
                X,
                prot,
                feature_names=["f0", "proxy", "f2"],
                n_permutations=8,
                random_state=1,
            )

        assert res.subsampled is True
        assert res.n_rows_supplied == 2000
        assert res.n == 200
        assert any("sample estimates" in note for note in res.notes)
        assert any("max_rows" in str(w.message) for w in caught)

    def test_a_dataset_inside_the_budget_is_untouched(self):
        """OVER-CORRECTION CONTROL: no sampling, no warning, no caveat note."""
        X, prot = _head_biased_dataset(n=150)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fairness_decomposition(
                _predict_from_proxy,
                X,
                prot,
                feature_names=["f0", "proxy", "f2"],
                n_permutations=8,
                random_state=1,
            )

        assert res.subsampled is False
        assert res.n == 150 and res.n_rows_supplied == 150
        assert not any("sample estimates" in note for note in res.notes)
        assert not any("max_rows" in str(w.message) for w in caught)

    def test_the_budget_cannot_delete_a_small_group(self):
        """The sample is STRATIFIED, so a 6-row stratum in 2000 rows survives a
        200-row budget. A plain random draw would lose it about 53% of the time,
        and losing it turns a disparity into a missing comparison."""
        rng = np.random.default_rng(8)
        n = 2000
        prot = np.array(["A"] * 997 + ["B"] * 997 + ["C"] * 6, dtype=object)
        X = np.column_stack([rng.normal(0, 1, n), (prot == "A").astype(float)])
        res = _quiet(
            fairness_decomposition,
            lambda Z: np.clip(0.5 + 0.4 * np.asarray(Z, float)[:, 1], 0, 1),
            X,
            prot,
            feature_names=["f0", "proxy"],
            n_permutations=4,
            random_state=1,
        )
        assert res.subsampled is True
        assert res.n <= 210


class TestTheDecompositionRefusesWhatItCannotMeasure:
    def test_grade_of_nan_is_not_the_all_clear(self):
        assert _grade(float("nan")) == "not_assessable"
        # OVER-CORRECTION CONTROL: the real bands are untouched.
        assert _grade(0.0) == "info"
        assert _grade(0.03) == "low"
        assert _grade(0.07) == "medium"
        assert _grade(0.12) == "high"
        assert _grade(0.30) == "critical"

    def test_an_unscorable_model_is_refused_not_ranked(self):
        """REFUSAL PIN. Measured before the fix: severity 'info',
        group_advantaged == group_disadvantaged == 'A', and the interpretation
        "No material demographic parity disparity between groups 'A' and 'A'."""
        rng = np.random.default_rng(5)
        prot = np.array(["A"] * 100 + ["B"] * 100, dtype=object)
        X = rng.normal(0, 1, (200, 3))

        with pytest.raises(ValueError) as excinfo:
            _quiet(
                fairness_decomposition,
                lambda Z: np.full(np.asarray(Z).shape[0], np.nan),
                X,
                prot,
                feature_names=["a", "b", "c"],
                n_permutations=4,
                random_state=1,
            )
        assert "COULD NOT CHECK" in str(excinfo.value)
        assert "not a finding of parity" in str(excinfo.value)

    def test_the_additivity_guarantee_is_stated_only_where_it_was_checked(self):
        """The guarantee used to be asserted unconditionally, and the only thing
        that could contradict it was `abs(residual) > 1e-6`, which is False for
        a NaN residual. So the note stood on a decomposition where nothing had
        been verified.

        The wording now names the residual it checked, and
        `additivity_verified` is three-state: True (checked and holds), False
        (checked and exceeded), None (could not check). A run that says
        "verified" and cannot show the number is the defect returning.

        The NaN-residual branch itself is DEFENCE IN DEPTH and is not reachable
        through the public entry point any more: `shapley_matrix` derives
        `predictions` from the attribution matrix
        (``predictions = base_value + phi.sum(axis=1)``), so a NaN contribution
        makes the group means NaN too, and the unrankable-group guard pinned in
        the test above refuses first. Two levels, and the test above is the one
        that exercises the live path.
        """
        X, prot = _head_biased_dataset(n=150)
        res = _quiet(
            fairness_decomposition,
            _predict_from_proxy,
            X,
            prot,
            feature_names=["f0", "proxy", "f2"],
            n_permutations=8,
            random_state=1,
        )

        assert res.additivity_verified is True
        assert math.isfinite(res.residual) and abs(res.residual) <= 1e-6
        verified = [n for n in res.notes if "contributions sum to the" in n]
        assert len(verified) == 1
        assert "verified, residual" in verified[0], (
            "the additivity guarantee is stated without naming the residual it "
            "was checked against: " + verified[0]
        )

    def test_a_real_decomposition_still_reads_exactly_as_a_real_one(self):
        """OVER-CORRECTION CONTROL."""
        rng = np.random.default_rng(0)
        n = 240
        prot = np.array(rng.choice(["A", "B"], size=n).tolist(), dtype=object)
        proxy = np.where(prot == "A", rng.normal(1.0, 0.3, n), rng.normal(-1.0, 0.3, n))
        X = np.column_stack([proxy, rng.normal(0, 1, n)])
        res = _quiet(
            fairness_decomposition,
            lambda Z: 1.0 / (1.0 + np.exp(-2.0 * np.asarray(Z, float)[:, 0])),
            X,
            prot,
            feature_names=["proxy", "neutral"],
            n_permutations=20,
            random_state=42,
        )

        assert res.group_advantaged == "A"
        assert res.group_disadvantaged == "B"
        assert res.total_disparity > 0.05
        assert res.severity in ("medium", "high", "critical")
        assert res.flagged_proxies == ["proxy"]
        assert res.additivity_verified is True


# ===========================================================================
# 7. recourse: a GDPR Article 22 sentence for a model nobody could score
# ===========================================================================

_ROBUST_SENTENCE = (
    "No counterfactual within the search budget flipped the decision; "
    "the outcome is robust to small input changes, or recourse needs "
    "changes beyond the observed data ranges."
)


def _bg(seed=0, n=300):
    return np.random.default_rng(seed).uniform(-3, 3, size=(n, 3))


class TestRecourseDoesNotClaimRobustnessItNeverMeasured:
    def test_an_unscorable_model_is_could_not_check(self):
        """REFUSAL PIN. Measured before the fix, the note was BYTE-IDENTICAL to
        the one a genuinely robust, fully scored refusal produces."""
        bg = _bg()
        res = _quiet(
            generate_recourse,
            lambda Z: np.full(np.asarray(Z).shape[0], np.nan),
            bg[0].copy(),
            bg,
            n_candidates=400,
            random_state=1,
        )

        assert res.found is None
        assert res.n_scorable_candidates == 0
        assert _ROBUST_SENTENCE not in " ".join(res.notes)
        joined = " ".join(res.notes)
        assert "COULD NOT CHECK" in joined
        assert "NOT a finding that the outcome is robust" in joined

    def test_a_model_that_scores_the_row_but_no_candidate_is_also_refused(self):
        """The other half: the individual is scorable, the search is not."""
        bg = _bg(seed=2)

        def only_one_row(Z):
            Z = np.asarray(Z, dtype=float)
            return np.full(Z.shape[0], 0.02 if Z.shape[0] == 1 else np.nan)

        res = _quiet(
            generate_recourse, only_one_row, bg[0].copy(), bg, n_candidates=300, random_state=1
        )

        assert res.found is None
        assert res.n_scorable_candidates == 0
        assert res.prediction_before == pytest.approx(0.02)
        assert _ROBUST_SENTENCE not in " ".join(res.notes)

    def test_a_genuinely_robust_refusal_still_says_so(self):
        """OVER-CORRECTION CONTROL. A model that IS scored, 400 times, and never
        flips, keeps the exact sentence it earned."""
        bg = _bg(seed=1)
        res = _quiet(
            generate_recourse,
            lambda Z: np.full(np.asarray(Z).shape[0], 0.02),
            bg[0].copy(),
            bg,
            n_candidates=400,
            random_state=1,
        )

        assert res.found is False
        assert res.n_scorable_candidates == 400
        assert _ROBUST_SENTENCE in res.notes

    def test_a_real_search_still_finds_and_reports_recourse(self):
        """OVER-CORRECTION CONTROL."""
        bg = _bg(seed=3)

        def predict(Z):
            Z = np.asarray(Z, dtype=float)
            return 1.0 / (1.0 + np.exp(-(1.5 * Z[:, 0] - 1.5 * Z[:, 1])))

        res = _quiet(
            generate_recourse,
            predict,
            np.array([-2.0, 2.0, 0.0]),
            bg,
            feature_names=["income", "debt", "noise"],
            random_state=1,
        )

        assert res.found is True
        assert res.counterfactuals
        assert res.n_scorable_candidates == res.n_candidates
        for cf in res.counterfactuals:
            assert cf.prediction_after >= 0.5
            assert cf.sentence.endswith("approved.")

    def test_an_unobserved_feature_is_not_given_a_fabricated_zero(self):
        assert np.all(np.isnan(_column_sampler(np.full(50, np.nan), np.random.default_rng(0), 5)))
        # OVER-CORRECTION CONTROL: an ordinary column still samples real values.
        col = np.array([1.0, 2.0, 3.0, np.nan])
        drawn = _column_sampler(col, np.random.default_rng(0), 20)
        assert set(np.unique(drawn)).issubset({1.0, 2.0, 3.0})

    def test_an_all_nan_feature_is_excluded_from_the_search_and_named(self):
        bg = _bg(seed=4)
        bg[:, 2] = np.nan

        def predict(Z):
            Z = np.asarray(Z, dtype=float)
            return 1.0 / (1.0 + np.exp(-(1.5 * Z[:, 0] - 1.5 * Z[:, 1])))

        res = _quiet(
            generate_recourse,
            predict,
            np.array([-2.0, 2.0, 0.0]),
            bg,
            feature_names=["income", "debt", "unrecorded"],
            random_state=1,
        )

        assert any("unrecorded" in note for note in res.notes)
        for cf in res.counterfactuals:
            assert all(ch.feature != "unrecorded" for ch in cf.changes)


# ===========================================================================
# 8. explainer: `nan >= 0.05` is False, so the NaN case fell to the fallback
# ===========================================================================


def _experiment(overall_p):
    return ExperimentResult(
        overall_effect=0.0123,
        overall_ci=(0.001, 0.02),
        overall_p_value=overall_p,
        intersection_effects=[],
        heterogeneity_detected=False,
        heterogeneity_p_value=0.42,
        n_intersections=3,
        design_type=DesignType.SIMPLE_AB,
    )


class TestAnUnrunOverallTestIsNotAConsistentResult:
    def test_the_nan_case_no_longer_reads_as_a_strong_real_effect(self):
        """REFUSAL PIN. Measured before the fix:

            measured p=0.90    sev=info ['Overall effect not significant...']
            measured p=0.001   sev=low  ['Results look consistent...']
            NOT MEASURED p=nan sev=info ['Results look consistent...']

        The producer sets that NaN deliberately (both arms constant, no t
        statistic), and the unmeasured reading was byte-identical to the strong
        real effect."""
        out = _explain_experiment_result(_experiment(float("nan")))
        strong = _explain_experiment_result(_experiment(0.001))

        assert out.recommendations != strong.recommendations
        assert "Results look consistent. Proceed with deployment review." not in out.recommendations
        assert any("COULD NOT CHECK" in r for r in out.recommendations)
        assert out.severity == "medium"
        assert "significance: not assessed" in out.summary

        card = [e for e in out.explanations if e.metric_name == "Overall Treatment Effect"][0]
        assert "COULD NOT CHECK" in card.evaluation
        assert "p = nan" not in card.evaluation
        assert card.severity == "medium"

    @pytest.mark.parametrize(
        "p,severity,expected_rec",
        [
            (0.90, "info", "Overall effect not significant. Consider extending the experiment."),
            (0.001, "low", "Results look consistent. Proceed with deployment review."),
        ],
    )
    def test_a_measured_p_reads_exactly_as_before(self, p, severity, expected_rec):
        """OVER-CORRECTION CONTROL, byte for byte."""
        out = _explain_experiment_result(_experiment(p))

        assert out.severity == severity
        assert out.recommendations == [expected_rec]
        assert "COULD NOT CHECK" not in " ".join(out.recommendations)
        assert out.summary == (
            f"Overall effect +0.0123 (p={p:.4f}). 3 intersections analysed. Heterogeneity: no."
        )


# ===========================================================================
# 9. _statistics: the two lower-severity holes
# ===========================================================================


class TestNoCorrectionStillCountsWhatDidNotRun:
    def test_method_none_agrees_with_the_other_two(self):
        """REFUSAL PIN. Measured before the fix on [0.001, nan, 0.9]:
        method='none' answered tested_mask=None, n_not_tested=0 and warned
        nothing, while bonferroni and fdr answered [True, False, True], 1 and
        warned."""
        family = np.array([0.001, np.nan, 0.9])
        results = {}
        for method in ("bonferroni", "fdr", "none"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results[method] = (
                    apply_multiple_testing_correction(family, method=method),
                    len(caught),
                )

        for method, (res, n_warn) in results.items():
            assert res.tested_mask is not None, method
            assert res.tested_mask.tolist() == [True, False, True], method
            assert res.n_not_tested == 1, method
            assert n_warn >= 1, method
            assert not bool(res.rejection_mask[1]), method
            assert math.isnan(float(res.adjusted_p_values[1])), method

    def test_an_uncorrected_family_of_real_p_values_is_unchanged(self):
        """OVER-CORRECTION CONTROL: 'none' still means no correction."""
        family = np.array([0.01, 0.02, 0.03])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = apply_multiple_testing_correction(family, method="none")

        assert res.method == "none"
        assert np.allclose(res.adjusted_p_values, family)
        assert res.rejection_mask.tolist() == [True, True, True]
        assert res.n_rejected == 3
        assert res.n_not_tested == 0
        assert not caught


class TestTheAttainedPermutationResolutionIsAlwaysRecorded:
    @staticmethod
    def _data(n=400):
        rng = np.random.default_rng(0)
        attr = np.array(["A"] * (n // 2) + ["B"] * (n // 2), dtype=object)
        return rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n), attr

    def test_nan_attrition_is_disclosed_even_when_the_cap_did_not_bind(self):
        """REFUSAL PIN. Measured before the fix at confidence_level=0.999
        (alpha 0.001, design 4000 permutations, MAX_PERMUTATIONS not reached):
        200 usable permutations, floor 0.004975 against a 0.0005 bar, and the
        metadata carried neither flag."""
        y_true, y_pred, attr = self._data()
        state = {"i": 0}

        def flaky(yt, yp, sa, **kw):
            state["i"] += 1
            if state["i"] > 1 and state["i"] % 20 != 0:
                return float("nan")
            groups = np.unique(sa)
            rates = [float(np.mean(yp[sa == g])) for g in groups]
            return max(rates) - min(rates)

        md = _quiet(
            compute_metric_with_ci,
            flaky,
            y_true,
            y_pred,
            attr,
            n_bootstrap=300,
            confidence_level=0.999,
            random_state=1,
        ).metadata

        assert md["n_permutation"] < md["n_permutation_requested"]
        assert md["min_resolvable_p_value"] == pytest.approx(1.0 / (md["n_permutation"] + 1))
        assert md["min_resolvable_p_value"] > 0.001 / 2.0
        assert md["permutation_resolution_limited"] is True
        assert "NOT DETECTABLE" in md["permutation_design_note"]

    def test_an_ordinary_run_records_the_floor_without_raising_the_flag(self):
        """OVER-CORRECTION CONTROL. A run with the resolution its design asked
        for must not be labelled limited, or the flag means nothing."""
        y_true, y_pred, attr = self._data()

        def dp(yt, yp, sa, **kw):
            groups = np.unique(sa)
            rates = [float(np.mean(yp[sa == g])) for g in groups]
            return max(rates) - min(rates)

        md = _quiet(
            compute_metric_with_ci,
            dp,
            y_true,
            y_pred,
            attr,
            n_bootstrap=300,
            confidence_level=0.95,
            random_state=1,
        ).metadata

        assert md["n_permutation"] == 199
        assert md["min_resolvable_p_value"] == pytest.approx(1 / 200)
        assert md["min_resolvable_p_value"] <= 0.05 / 2.0
        assert "permutation_resolution_limited" not in md
        assert "permutation_design_note" not in md
