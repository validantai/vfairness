"""Beta Go-Live Stage 2, group s2g03: three states at six public entries.

Every test here asserts at a PUBLIC entry point, and every defect class has a
CONTROL beside it showing that healthy data still produces the correct
measurement. A fix that makes everything refuse is a worse defect than the one
it replaces, and it passes any test that only checks the degenerate case.

The six capabilities:

    feature_engineering_analysis  FeatureEngineeringAnalyzer.full_analysis
    power_analysis                FairnessPowerAnalyzer.*
    counterfactual_tester         CounterfactualTester.run_test
    tool_bias_auditor             ToolBiasAuditor.analyze_tool_calls
    find_feasible_thresholds      find_feasible_thresholds
    group_threshold               GroupThresholdOptimizer.fit / .predict
                                  compute_constraint_violation
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.llm.counterfactual as _cf_module
from vfairness import (
    CounterfactualTester,
    FeatureEngineeringAnalyzer,
    GroupThresholdOptimizer,
    ThresholdOptimizer,
    ToolBiasAuditor,
    compute_constraint_violation,
    create_threshold_optimizer,
    find_feasible_thresholds,
)
from vfairness.exceptions import ConfigurationError
from vfairness.operations.experimentation import FairnessExperiment, FairnessPowerAnalyzer


def _messages(records) -> str:
    return " || ".join(str(r.message) for r in records)


# ---------------------------------------------------------------------------
# 1. feature_engineering_analysis: an all-clear over a screen that never ran
# ---------------------------------------------------------------------------


def _proxy_frame(n: int, *, constant_attr: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    race = np.zeros(n, int) if constant_attr else (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame(
        {
            "race": race,
            "income": race * 50000 + rng.normal(0, 500, n),
            "noise": rng.normal(size=n),
        }
    )


class TestProxyScreenSaysWhenItDidNotLook:
    """Measured before the fix at n=50, where corr(income, race) is 0.9998:

        proxy_variables [], risk_summary all zeros, ZERO warnings, and the
        summary a reader sees ended "No significant proxy variables detected."

    The identical generator at n=500 reports that same feature CRITICAL.
    """

    def test_an_unscreened_attribute_is_not_an_all_clear(self):
        df = _proxy_frame(50)
        # Sanity: the fixture really does hold the proxy the screen misses.
        assert abs(np.corrcoef(df["income"], df["race"])[0, 1]) > 0.95

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = FeatureEngineeringAnalyzer(
                df, ["race"], feature_columns=["income", "noise"]
            ).full_analysis()

        assert report.proxy_screen_complete is False
        assert report.risk_summary["not_assessed"] == 1
        assert [e["protected_attribute"] for e in report.screens_not_run] == ["race"]
        assert "min_sample_size=100" in report.screens_not_run[0]["reason"]
        assert "No significant proxy variables detected." not in report.summary
        assert "COULD NOT CHECK" in report.summary
        assert "did not run" in _messages(caught)

    def test_a_nan_correlation_is_not_graded_negligible(self):
        """A constant protected attribute makes every correlation NaN. Each
        feature still came back graded NEGLIGIBLE, the SAFEST band, counted in
        risk_summary['negligible'], carrying evidence pvalue_status='computed'
        over pvalue=nan."""
        df = _proxy_frame(500, constant_attr=True)
        assert df["race"].nunique() == 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = FeatureEngineeringAnalyzer(
                df, ["race"], feature_columns=["income", "noise"]
            ).full_analysis()

        assert report.risk_summary["negligible"] == 0
        assert report.proxy_variables == []
        assert len(report.ungraded_proxy_screens) == 2
        # The count is the sum of both could-not-check channels. Asserted as an
        # invariant rather than a literal: the proxy-CHAIN search contributes to
        # screens_not_run as well on this same input, and pinning a total would
        # make this test a hostage to that separate surface.
        assert report.risk_summary["not_assessed"] == len(report.screens_not_run) + len(
            report.ungraded_proxy_screens
        )
        assert report.risk_summary["not_assessed"] >= 2
        for screen in report.ungraded_proxy_screens:
            assert math.isnan(screen.correlation)
            # The false status is worse than a missing one.
            assert screen.evidence["pvalue_status"] == "could_not_compute"
            assert screen.evidence["risk_level_status"] == "not_assessed"
        assert "NOT GRADED" in report.summary
        assert "no measurable correlation" in _messages(caught)

    def test_the_third_state_reaches_to_dict(self):
        report = FeatureEngineeringAnalyzer(
            _proxy_frame(50), ["race"], feature_columns=["income", "noise"]
        ).full_analysis()
        payload = report.to_dict()
        assert payload["proxy_screen_complete"] is False
        assert len(payload["screens_not_run"]) == 1

    def test_control_healthy_data_still_finds_the_critical_proxy(self):
        """OVER-CORRECTION CONTROL. The same generator at n=500 must still
        report income CRITICAL, with nothing routed to not_assessed and no
        warning at all."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = FeatureEngineeringAnalyzer(
                _proxy_frame(500), ["race"], feature_columns=["income", "noise"]
            ).full_analysis()

        assert report.proxy_screen_complete is True
        assert report.risk_summary["critical"] == 1
        assert report.risk_summary["not_assessed"] == 0
        assert report.high_risk_features == ["income"]
        assert report.proxy_variables[0].evidence["pvalue_status"] == "computed"
        assert "COULD NOT CHECK" not in report.summary
        assert "did not run" not in _messages(caught)


# ---------------------------------------------------------------------------
# 2. power_analysis: an answer invariant to whether any observation exists
# ---------------------------------------------------------------------------


def _power_experiment(kind: str, per_cell: int = 60) -> FairnessExperiment:
    rows = [{"gender": g, "race": r} for g in "FM" for r in "BW" for _ in range(per_cell)]
    df = pd.DataFrame(rows)
    if kind == "real":
        df["approved"] = np.random.default_rng(0).integers(0, 2, len(df)).astype(float)
    elif kind == "none":
        df["approved"] = np.nan
    elif kind == "five":
        df["approved"] = np.nan
        df.loc[df.index[:5], "approved"] = 1.0
    else:  # pragma: no cover - guard against a typo in a future edit
        raise AssertionError(kind)
    return FairnessExperiment(
        control_data=df.copy(),
        treatment_data=df.copy(),
        protected_attributes=["gender", "race"],
        outcome_column="approved",
    )


class TestPowerIsAboutObservationsNotRows:
    """Measured before the fix on three datasets identical in SHAPE (60 rows
    per intersection per arm) and differing only in how many outcome values
    existed, 240 real / 5 real / 0 real. All three returned:

        power_for_sample_size(0.5) 0.7818, required_sample_size(0.5) 63,
        minimum_detectable_effect() 0.5115, plan.n_underpowered 4,
        not_assessed [], get_power_summary n_control 60, zero warnings.

    The answers were invariant to whether any observation existed at all.
    """

    def test_an_intersection_with_no_outcome_value_is_not_powered_at_n_60(self):
        analyzer = FairnessPowerAnalyzer(_power_experiment("none"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            power = analyzer.power_for_sample_size(effect_size=0.5)
            mde = analyzer.minimum_detectable_effect()
            summary = analyzer.get_power_summary(effect_size=0.5)

        key = ("F", "B")
        assert math.isnan(power[key])
        assert math.isnan(mde[key])
        row = summary[summary["intersection"] == str(key)].iloc[0]
        # Rows and observations are BOTH shown; either alone hides half of it.
        assert row["n_control"] == 0 and row["n_rows_control"] == 60
        assert row["is_powered"] is None
        assert math.isnan(row["power"])
        assert "no outcome observation" in _messages(caught)

    def test_the_sampling_plan_routes_it_to_not_assessed_not_underpowered(self):
        analyzer = FairnessPowerAnalyzer(_power_experiment("none"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=1000, effect_size=0.5)

        assert plan.n_underpowered == 0
        assert len(plan.not_assessed) == 4
        rationale = plan.rationale[("F", "B")]
        assert rationale.startswith("NOT ASSESSED")
        assert "60 control and 60 treatment row(s) are present but only 0" in rationale
        assert "already powered" not in plan.rationale.values()

    def test_ordinary_partial_missingness_separates_too(self):
        """The 5-of-240 case matters more than the all-empty one: ordinary
        missingness, not a contrived column, produced the same invariant
        answer. An intersection with 5 observations is genuinely underpowered
        (more data helps); one with 0 is unmeasured (more rows will not)."""
        analyzer = FairnessPowerAnalyzer(_power_experiment("five"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=1000, effect_size=0.5)
            summary = analyzer.get_power_summary(effect_size=0.5)

        assert plan.n_underpowered == 1
        assert len(plan.not_assessed) == 3
        assert "(have 5)" in plan.rationale[("F", "B")]
        assert set(summary["n_control"]) == {0, 5}

    def test_required_sample_size_refuses_a_zero_effect_size(self):
        """It used to return the int 999_999 under a warning that said
        'returning inf'. math.isinf on the returned value was False and 999999
        is a number a caller can budget against."""
        analyzer = FairnessPowerAnalyzer(_power_experiment("real"))
        with pytest.raises(ConfigurationError, match="effect of zero"):
            analyzer.required_sample_size(effect_size=0.0)

    def test_control_fully_observed_data_is_unchanged(self):
        """OVER-CORRECTION CONTROL. With every outcome present the numbers are
        exactly what they were before the fix, and nothing warns."""
        analyzer = FairnessPowerAnalyzer(_power_experiment("real"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            key = ("F", "B")
            assert analyzer.power_for_sample_size(effect_size=0.5)[key] == pytest.approx(0.7818)
            assert analyzer.required_sample_size(effect_size=0.5)[key] == 63
            assert analyzer.minimum_detectable_effect()[key] == pytest.approx(0.5115)
            plan = analyzer.adaptive_sampling_plan(budget=1000, effect_size=0.5)

        assert plan.n_underpowered == 4
        assert plan.not_assessed == []
        assert "no outcome observation" not in _messages(caught)


# ---------------------------------------------------------------------------
# 3. counterfactual_tester: a healthy variant concealing a silent one
# ---------------------------------------------------------------------------


class _OneVariantSilent:
    """Lakisha returns nothing; James and Jamal return scoreable text."""

    def send_prompt(self, prompt, system_prompt=None, **kwargs):
        if "Lakisha" in prompt:
            return {"text": ""}
        if "Jamal" in prompt:
            return {"text": "This is a bad terrible awful poor risky application and I decline it."}
        return {
            "text": "This is a good excellent wonderful great strong application and I approve it."
        }


class _AllVariantsAnswer(_OneVariantSilent):
    def send_prompt(self, prompt, system_prompt=None, **kwargs):
        if "Jamal" in prompt:
            return {"text": "This is a bad terrible awful poor risky application and I decline it."}
        return {
            "text": "This is a good excellent wonderful great strong application and I approve it."
        }


@pytest.fixture()
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(_cf_module._time, "sleep", lambda *a, **k: None)


class TestCounterfactualProvenanceAggregatesByWorst:
    """`data_quality` is a STRING, so the old `setdefault` kept whichever pair
    was seen FIRST and stamped it over the aggregate. Measured before the fix
    with three variants where James and Jamal were scored and Lakisha returned
    nothing at all:

        data_quality 'good', unscored_metrics [], a full set of numeric deltas,
        is_significant False, metadata n_tests_not_run 0

    while the significance_tests list named only Jamal. The IDENTICAL outage
    with two variants read honestly ('insufficient', every delta None,
    is_significant None), so adding a healthy variant CONCEALED the failure of
    another one.
    """

    def _run(self, proxy, variants):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = CounterfactualTester(proxy=proxy, n_runs=4).run_test(
                "Evaluate {name}.", {"name": variants}
            )
        return result, caught

    def test_a_silent_variant_is_not_hidden_by_a_healthy_one(self, _no_retry_sleep):
        result, caught = self._run(_OneVariantSilent(), ["James", "Jamal", "Lakisha"])
        metrics = result.disparity_metrics

        # Fixture check: the run really did produce one silent arm.
        silent = [v for v in result.variants if v["demographic"] == "Lakisha"][0]
        assert silent["empty_count"] == 4

        assert metrics["data_quality"] == "insufficient"
        assert metrics["n_pairs_not_compared"] == 1
        assert metrics["variants_not_compared"] == ["Lakisha"]
        assert metrics["disparity_is_lower_bound"] is True
        assert result.metadata.parameters["n_tests_not_run"] == 1
        assert "could not be compared" in _messages(caught)

    def test_every_variant_appears_in_the_significance_record(self, _no_retry_sleep):
        """n_tests_not_run=0 was a false statement about the run: the four
        Lakisha comparisons never happened and nothing recorded that."""
        result, _ = self._run(_OneVariantSilent(), ["James", "Jamal", "Lakisha"])
        tests = result.metadata.parameters["significance_tests"]
        named = {str(t["variant"]) for t in tests}
        assert named == {"Jamal", "Lakisha"}
        not_run = [t for t in tests if not t["tested"]]
        assert [t["variant"] for t in not_run] == ["Lakisha"]
        assert "never compared" in not_run[0]["reason"]

    def test_when_no_pair_was_compared_no_delta_stands(self, _no_retry_sleep):
        result, _ = self._run(_OneVariantSilent(), ["James", "Lakisha"])
        metrics = result.disparity_metrics
        assert metrics["n_pairs_compared"] == 0
        assert metrics["sentiment_delta"] is None
        assert result.is_significant is None
        assert set(metrics["unscored_metrics"]) >= {"sentiment_delta", "length_delta"}

    def test_control_three_answering_variants_read_clean(self, _no_retry_sleep):
        """OVER-CORRECTION CONTROL. With every variant answering, the run is
        'good', nothing is listed as uncompared and no warning fires."""
        result, caught = self._run(_AllVariantsAnswer(), ["James", "Jamal", "Lakisha"])
        metrics = result.disparity_metrics
        assert metrics["data_quality"] == "good"
        assert metrics["n_pairs_compared"] == 2
        assert metrics["n_pairs_not_compared"] == 0
        assert metrics["disparity_is_lower_bound"] is False
        assert metrics["sentiment_delta"] == pytest.approx(2.0)
        assert result.metadata.parameters["n_tests_not_run"] == 0
        assert "could not be compared" not in _messages(caught)


# ---------------------------------------------------------------------------
# 4. tool_bias_auditor: a bootstrap interval built out of 0/0 draws
# ---------------------------------------------------------------------------


class TestToolBiasIntervalExcludesUndefinedDraws:
    """Measured before the fix on 198 approve + 2 escalate_to_human against
    200 approve, seed 42:

        disparity_ratio 0.0 (total exclusion of group B, a real measurement),
        confidence_interval (0.0, 1.0), no warning at all.

    133 of the 1000 draws had the tool in NEITHER resample and each was scored
    1.0; the distinct ratio values in the whole distribution were exactly
    {0.0, 1.0}, so the 97.5th-percentile upper bound of 1.0, PERFECT PARITY,
    was made entirely of those undefined draws.
    """

    def _rare_tool_result(self):
        traces_a = [{"tool": "approve"}] * 198 + [{"tool": "escalate_to_human"}] * 2
        traces_b = [{"tool": "approve"}] * 200
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = ToolBiasAuditor(random_seed=42).analyze_tool_calls(traces_a, traces_b)
        rare = next(r for r in results if r.tool_name == "escalate_to_human")
        return rare, caught

    def test_the_upper_bound_is_no_longer_perfect_parity(self):
        rare, caught = self._rare_tool_result()
        # The point estimate is a real measurement and must survive.
        assert rare.disparity_ratio == 0.0
        assert rare.confidence_interval[1] < 1.0
        assert rare.confidence_interval == (0.0, 0.0)
        assert rare.n_bootstrap_undefined == 133
        assert rare.ci_coverage == pytest.approx(0.867)
        assert "not scored as 1.0" in _messages(caught)

    def test_the_coverage_survives_serialisation(self):
        rare, _ = self._rare_tool_result()
        payload = rare.to_dict()
        # SerializableMixin turns the tuple into a list on the way out.
        assert payload["confidence_interval"] == [0.0, 0.0]
        assert payload["n_bootstrap_undefined"] == 133
        assert payload["ci_coverage"] == pytest.approx(0.867)

    def test_control_a_common_tool_is_unchanged(self):
        """OVER-CORRECTION CONTROL. A tool both groups use has no undefined
        draw, keeps its interval and warns about nothing."""
        traces_a = [{"tool": "approve"}] * 100 + [{"tool": "escalate_to_human"}] * 100
        traces_b = [{"tool": "approve"}] * 150 + [{"tool": "escalate_to_human"}] * 50
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = ToolBiasAuditor(random_seed=42).analyze_tool_calls(traces_a, traces_b)
        hit = next(r for r in results if r.tool_name == "escalate_to_human")
        assert hit.disparity_ratio == pytest.approx(0.5)
        assert hit.n_bootstrap_undefined == 0
        assert hit.ci_coverage == 1.0
        assert hit.confidence_interval[0] == pytest.approx(0.3773, abs=1e-3)
        assert hit.confidence_interval[1] == pytest.approx(0.6484, abs=1e-3)
        assert "not scored as 1.0" not in _messages(caught)


# ---------------------------------------------------------------------------
# 5. find_feasible_thresholds: feasibility for a metric that does not exist
# ---------------------------------------------------------------------------


def _threshold_data(seed: int = 5, n: int = 400):
    rng = np.random.default_rng(seed)
    sens = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    prob = rng.random(n)
    healthy = (rng.random(n) < 0.5).astype(int)
    return healthy, prob, sens


class TestFeasibilityRefusesAnUndefinedRate:
    """An undefined FPR became 0.0, which is the BEST attainable FPR, so the
    group matched every partner within tolerance. Measured before the fix with
    every y_true == 1 (no negatives anywhere, FPR is 0/0):

        false_positive_parity -> {'a': 100, 'b': 100} feasible thresholds
        equalized_odds        -> {'a': 100, 'b': 100}        warnings: []

    and with only group b lacking negatives, false_positive_parity gave
    {'a': 6, 'b': 100}: b feasible at all 100 thresholds against a rate
    measured nowhere, and group a's own answer driven from 100 down to 6 by
    that fabricated partner.
    """

    def test_no_negatives_anywhere_is_not_a_feasible_region(self):
        _, prob, sens = _threshold_data()
        y_all_positive = np.ones(len(prob), int)
        assert (y_all_positive == 0).sum() == 0  # fixture: FPR really is 0/0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_feasible_thresholds(
                y_all_positive, prob, sens, constraint="false_positive_parity"
            )

        assert dict(result) == {}
        assert set(result.not_assessable) == {"a", "b"}
        assert "0 negative label(s)" in result.not_assessable["a"]
        assert result.all_groups_assessed is False
        assert "could NOT be assessed" in _messages(caught)

    def test_an_unassessed_group_is_a_keyerror_not_an_empty_list(self):
        """An empty list is a MEASUREMENT ('no threshold works'). A group whose
        metric does not exist must not be indexable as one."""
        _, prob, sens = _threshold_data()
        result = find_feasible_thresholds(
            np.ones(len(prob), int), prob, sens, constraint="false_positive_parity"
        )
        with pytest.raises(KeyError, match="NOT assessed"):
            result["a"]

    def test_one_broken_group_does_not_drive_its_partners_answer(self):
        """The cross-contamination is the decisive part: a's feasible count
        moved from 100 to 6 purely because of a rate measured nowhere in b."""
        healthy, prob, sens = _threshold_data()
        y = healthy.copy()
        y[len(y) // 2 :] = 1  # group b loses every negative

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = find_feasible_thresholds(y, prob, sens, constraint="false_positive_parity")

        assert "b" in result.not_assessable
        assert "a" in result.not_assessable
        assert "at least 2" in result.not_assessable["a"]
        assert dict(result) == {}

    def test_a_single_group_is_not_feasible_everywhere(self):
        healthy, prob, sens = _threshold_data()
        half = len(prob) // 2
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = find_feasible_thresholds(
                healthy[:half], prob[:half], sens[:half], constraint="predictive_parity"
            )
        assert dict(result) == {}
        assert "at least 2" in result.not_assessable["a"]

    def test_control_healthy_data_keeps_its_feasible_sets(self):
        """OVER-CORRECTION CONTROL. The counts recorded before the fix on
        healthy data: 100/100 for false_positive_parity and 73/75 for
        equalized_odds, with nothing refused and no warning."""
        healthy, prob, sens = _threshold_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fpp = find_feasible_thresholds(healthy, prob, sens, constraint="false_positive_parity")
            eo = find_feasible_thresholds(healthy, prob, sens, constraint="equalized_odds")

        assert {k: len(v) for k, v in fpp.items()} == {"a": 100, "b": 100}
        assert {k: len(v) for k, v in eo.items()} == {"a": 73, "b": 75}
        assert fpp.all_groups_assessed and eo.all_groups_assessed
        assert "could NOT be assessed" not in _messages(caught)


# ---------------------------------------------------------------------------
# 6. compute_constraint_violation: half a constraint graded as the whole
# ---------------------------------------------------------------------------


class TestEqualizedOddsNeedsBothArms:
    """max(a, b) returns its FIRST operand when the comparison against a NaN is
    False, so an equalized-odds verdict silently became the TPR verdict alone.
    Measured before the fix with no negatives anywhere and a TPR gap of 0.02:

        violation 0.02, is_satisfied True, warnings []
        group_metrics {'A': {'tpr': 0.9, 'fpr': nan},
                       'B': {'tpr': 0.88, 'fpr': nan}}

    while the IDENTICAL fpr values one constraint name away
    (false_positive_parity) returned NaN / None with a warning.
    """

    def _one_label_case(self):
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = np.ones(200, dtype=int)
        y_pred = np.zeros(200, dtype=int)
        y_pred[:90] = 1
        y_pred[100:188] = 1
        return y_true, y_pred, sens

    def test_an_unmeasurable_arm_makes_the_conjunction_unmeasurable(self):
        y_true, y_pred, sens = self._one_label_case()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_constraint_violation(
                y_true, y_pred, sens, constraint="equalized_odds", tolerance=0.05
            )

        assert math.isnan(result.violation)
        assert result.is_satisfied is None
        # The measured half is NOT deleted: the TPR values still stand.
        assert result.group_metrics["A"]["tpr"] == pytest.approx(0.9)
        assert math.isnan(result.group_metrics["A"]["fpr"])
        assert "could not be evaluated" in _messages(caught)

    def test_the_library_no_longer_contradicts_itself_one_name_away(self):
        y_true, y_pred, sens = self._one_label_case()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            eo = compute_constraint_violation(
                y_true, y_pred, sens, constraint="equalized_odds", tolerance=0.05
            )
            fpp = compute_constraint_violation(
                y_true, y_pred, sens, constraint="false_positive_parity", tolerance=0.05
            )
        assert eo.is_satisfied is fpp.is_satisfied is None

    def test_the_factory_built_optimizer_is_not_feasible_either(self):
        """RE-AIMED 2026-09-17 (BGL-S2b audit). SUBJECT UNCHANGED: the factory
        built optimizer must not hand back a fabricated feasible verdict when
        one arm of equalized odds cannot be measured.

        The old expectation was ``is_feasible is None``. That was wrong in the
        other direction, and the audit called it deleting a real finding: the
        FPR arm is dead here, but the TPR arm IS measured, and at the selected
        thresholds its gap is 0.0600 against a tolerance of 0.0500. No value of
        the unmeasured arm can bring a max of the two back inside tolerance, so
        the constraint is determinately VIOLATED and refusing to say so threw
        away a true finding. None is now reserved for the case where the
        measured arm is INSIDE tolerance and the unknown arm really could
        decide it, which is what the first test in this class pins.
        """
        sens = np.array(["A"] * 100 + ["B"] * 100)
        prob = np.random.default_rng(1).random(200)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = create_threshold_optimizer("group", "equalized_odds", tolerance=0.05)
            opt.fit(y_true=np.ones(200, dtype=int), y_prob=prob, sensitive_attr=sens)
        assert opt.result_.is_feasible is False
        assert "Feasible: True" not in opt.result_.summary()
        # and the half that was NOT measured is disclosed, not buried
        violation = opt.result_.constraint_violations[0]
        assert violation.violation_is_lower_bound is True
        assert violation.unmeasured == ("A.fpr", "B.fpr")
        assert "lower bound" in opt.result_.summary()

    def test_control_a_measurable_equalized_odds_still_grades(self):
        """OVER-CORRECTION CONTROL. With both arms measurable the verdict is a
        real bool and the violation is the larger of the two spreads."""
        rng = np.random.default_rng(0)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_constraint_violation(
                y_true, y_pred, sens, constraint="equalized_odds", tolerance=0.05
            )
        assert isinstance(result.is_satisfied, bool)
        assert math.isfinite(result.violation)
        assert result.violation == pytest.approx(0.101, abs=1e-3)
        assert "could not be evaluated" not in _messages(caught)


# ---------------------------------------------------------------------------
# 7. group_threshold: an "optimum" that is a scan-order artifact
# ---------------------------------------------------------------------------


class TestThresholdSearchSaysWhenTheObjectiveDecidedNothing:
    """When the objective cannot be computed it is NaN, every '>' comparison is
    False, and the search degrades to "keep the first feasible grid point".
    Measured before the fix on 200 rows whose y_true was all ones:

        group_thresholds {'a': 0.05, 'b': 0.05}
        optimization_details {'objective': 'balanced_accuracy', ...}

    and scanning the SAME grid in reverse returned {'a': 0.95, 'b': 0.95} on
    identical data, while the healthy control was order-invariant.
    """

    def _degenerate_fit(self, n_thresholds=20):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 200)
        sens = np.array(["a"] * 100 + ["b"] * 100)
        y_true = np.ones(200, int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = GroupThresholdOptimizer(n_thresholds=n_thresholds)
            opt.fit(y_true=y_true, y_prob=prob, sensitive_attr=sens)
        return opt, caught

    def test_the_artifact_admits_the_objective_measured_nothing(self):
        opt, caught = self._degenerate_fit()
        details = opt.result_.optimization_details
        # Fixture check: the search really did visit candidates.
        assert details["n_objective_candidates"] == 400
        assert details["n_objective_measured"] == 0
        assert details["objective_measured"] is False
        assert "are NOT an optimum" in _messages(caught)

    def test_the_summary_a_reader_sees_carries_it(self):
        """The flag used to live only in optimization_details, which summary()
        never printed, so it reached no reader."""
        opt, _ = self._degenerate_fit()
        text = opt.result_.summary()
        assert "COULD NOT CHECK" in text
        assert "NOT an optimum" in text

    def test_the_single_threshold_sibling_discloses_it_too(self):
        """Fixing one branch moves the defect to its sibling: the same latch
        exists in ThresholdOptimizer.fit and in the >2-group path."""
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = ThresholdOptimizer(objective="balanced_accuracy", n_thresholds=20)
            opt.fit(
                y_true=np.ones(200, int),
                y_prob=rng.uniform(0, 1, 200),
                sensitive_attr=np.array(["a"] * 100 + ["b"] * 100),
            )
        assert opt.result_.optimization_details["objective_measured"] is False

    def test_an_unmeasured_incumbent_cannot_outrank_a_measured_candidate(self):
        """The mixed case. `best_objective` used to be assigned
        unconditionally, so an unmeasurable first candidate put NaN into it and
        every later candidate with a REAL objective then lost
        `measured > nan` (False) and could never be selected."""
        from vfairness.post_processing.threshold_optimization.optimizer import _ObjectiveScan

        scan = _ObjectiveScan()
        assert scan.observe(float("nan")) is False
        assert scan.observe(0.7) is True
        assert scan.any_measured is True
        assert scan.details()["n_objective_measured"] == 1
        assert scan.details()["n_objective_candidates"] == 2

    def test_control_healthy_data_still_optimises_and_is_order_invariant(self):
        """OVER-CORRECTION CONTROL."""
        rng = np.random.default_rng(3)
        n = 400
        prob = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
        sens = np.array(["a", "b"] * (n // 2))
        y_true = (rng.random(n) < prob).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = GroupThresholdOptimizer(n_thresholds=20)
            opt.fit(y_true=y_true, y_prob=prob, sensitive_attr=sens)
        details = opt.result_.optimization_details
        assert details["objective_measured"] is True
        assert details["n_objective_measured"] == details["n_objective_candidates"] == 400
        assert "COULD NOT CHECK" not in opt.result_.summary()
        assert "are NOT an optimum" not in _messages(caught)


class TestUnscoreableRowsAreRefusedNotRejected:
    """Findings 7 and 8 of this group were already closed by Stage 1 (commit
    2b0fa20, ``_refuse_unscored_rows``). These pin the behaviour so a later
    change cannot quietly restore the silent rejection, and the sabotage run
    recorded in the report confirms they are load-bearing.

    Before that guard, a NaN score compared False against every threshold and
    was written 0, i.e. REJECTED, in a bare int array with no field to say so.
    """

    def _fitted(self):
        rng = np.random.default_rng(14)
        n = 200
        groups = np.array(["A", "B"] * (n // 2))
        prob = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
        y_true = (rng.random(n) < prob).astype(int)
        return GroupThresholdOptimizer().fit(y_true=y_true, y_prob=prob, sensitive_attr=groups)

    def test_predict_refuses_an_all_nan_batch(self):
        opt = self._fitted()
        with pytest.raises(ValueError, match="no usable score"):
            opt.predict(np.full(6, np.nan), np.array(["A", "B"] * 3))

    def test_predict_refuses_a_mixed_batch_rather_than_zeroing_the_nan_rows(self):
        opt = self._fitted()
        with pytest.raises(ValueError, match="2 of 6 row"):
            opt.predict(
                np.array([0.9, np.nan, 0.1, 0.95, np.nan, 0.2]),
                np.array(["A", "A", "A", "B", "B", "B"]),
            )

    def test_fit_predict_refuses_too(self):
        rng = np.random.default_rng(14)
        n = 200
        with pytest.raises(ValueError, match="no usable score"):
            GroupThresholdOptimizer().fit_predict(
                y_true=rng.integers(0, 2, n),
                y_prob=np.full(n, np.nan),
                sensitive_attr=np.array(["a", "b"] * (n // 2)),
            )

    def test_control_a_clean_batch_still_predicts(self):
        """OVER-CORRECTION CONTROL."""
        opt = self._fitted()
        out = opt.predict(np.array([0.9, 0.1, 0.95, 0.2]), np.array(["A", "A", "B", "B"]))
        assert out.tolist() == [1, 0, 1, 0]
