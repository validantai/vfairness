"""Audit 6, lane 4 (evaluation + experimentation), 2026-09-09.

The defect class: a value that was never measured is replaced by a neutral
default (0, 0.0, "already powered") and then graded, compared, ranked or
reported as if it were a measurement. The fix pattern, always: three states,
never two. An unmeasured value becomes NaN / None, is EXCLUDED from any
max/min/mean (never a 0.0 floor), a warning names what could not be measured,
and a derived verdict becomes None rather than False.

Every finding below was reproduced by execution BEFORE the fix; the measured
pre-fix values are quoted in the docstrings. Each finding is pinned twice: the
REFUSAL, and an OVER-CORRECTION CONTROL proving the healthy path still gives
its real answer with no warning.

  R-5     discovery.discover_intersectional_groups: an unassessable
          single-attribute disparity entered max() as a 0.0 floor. Mirror of
          audit finding H-01, closed in intersectional._generate_comparison.
  R-1     power.FairnessPowerAnalyzer.adaptive_sampling_plan: an intersection
          the budget never reached was reported "already powered" with
          additional_samples=0.
  PARETO  analysis.ExperimentAnalysis.compute_pareto_frontier: a variant
          missing a metric got 0 for it and came out Pareto-optimal.
  F18     _statistics.compute_metric_with_ci: 'bayesian' is documented but not
          implemented, and 'auto' named that branch.

F20 (the registry contract) is pinned in tests/test_manifest.py.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.evaluation.vfairness_metrics.intersectional as intersectional_module
from vfairness import demographic_parity_difference
from vfairness.evaluation.vfairness_metrics._statistics import (
    SMALL_SAMPLE_THRESHOLD,
    compute_metric_with_ci,
)
from vfairness.evaluation.vfairness_metrics.discovery import (
    discover_intersectional_groups,
    rank_fairness_issues,
)
from vfairness.exceptions import ConfigurationError
from vfairness.operations.experimentation.analysis import ExperimentAnalysis
from vfairness.operations.experimentation.experiment import FairnessExperiment
from vfairness.operations.experimentation.power import FairnessPowerAnalyzer


def _messages(records, needle: str):
    return [str(r.message) for r in records if needle in str(r.message)]


# ---------------------------------------------------------------------------
# R-5: discover_intersectional_groups, hidden disparity over unassessed attributes
# ---------------------------------------------------------------------------


def _gender_site_frame(constant_site: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 400
    gender = rng.choice(["F", "M"], size=n)
    site = np.array(["hq"] * n) if constant_site else rng.choice(["hq", "branch"], size=n)
    y_pred = np.where(gender == "M", rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
    return pd.DataFrame({"gender": gender, "site": site}), y_pred


_R5_KEYS = ("hidden_disparity", "reveals_more", "single_attributes_not_assessable")


class TestR5HiddenDisparityNeverUsesAZeroFloor:
    """Pre-fix, measured 2026-09-09: attributes=['gender','site'] with a
    constant 'site' (own disparity correctly nan) reported disparity=0.69,
    hidden_disparity=0, reveals_more=False and no warning from discovery.py;
    the nan rode through Python's max() on operand order and a raised
    exception was swallowed into single_disparities[attr] = 0.0."""

    @pytest.mark.parametrize("order", [["gender", "site"], ["site", "gender"]])
    def test_constant_attribute_is_named_and_excluded_from_the_maximum(self, order):
        df, y_pred = _gender_site_frame(constant_site=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = discover_intersectional_groups(df, order, y_pred, min_group_size=30)

        assert len(results) == 1
        r = results[0]
        assert r["single_attributes_not_assessable"] == ["site"]
        # 'gender' WAS assessed and the intersection is the same split, so the
        # comparison is real: nothing hidden, and False is the honest verdict.
        assert r["reveals_more"] is False
        assert r["hidden_disparity"] == 0
        named = _messages(w, "single-attribute disparity could not be assessed for 'site'")
        assert named, [str(x.message) for x in w]
        assert "excluded from every hidden-disparity comparison" in named[0]

    def test_a_raised_exception_is_recorded_with_its_reason_not_turned_into_zero(self, monkeypatch):
        df, y_pred = _gender_site_frame(constant_site=False)
        real = intersectional_module.identify_privileged_groups

        def site_fails(y_true, y_hat, attr, **kw):
            if set(np.asarray(attr).tolist()) <= {"hq", "branch"}:
                raise ValueError("simulated failure on site")
            return real(y_true, y_hat, attr, **kw)

        monkeypatch.setattr(intersectional_module, "identify_privileged_groups", site_fails)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = discover_intersectional_groups(
                df, ["gender", "site"], y_pred, min_group_size=30
            )

        assert len(results) == 1
        r = results[0]
        assert r["single_attributes_not_assessable"] == ["site"]
        # The comparison ran over 'gender' alone and is still a real number.
        assert math.isfinite(r["hidden_disparity"])
        assert r["reveals_more"] in (True, False)
        named = _messages(w, "could not be assessed for 'site'")
        assert named and "ValueError: simulated failure on site" in named[0]

    def test_no_assessable_single_attribute_yields_nan_and_none(self, monkeypatch):
        """Nothing to subtract from: three states, so the comparison is
        'could not check', never hidden_disparity=0 / reveals_more=False."""
        rng = np.random.default_rng(0)
        n = 400
        gender = rng.choice(["F", "M"], size=n)
        site = rng.choice(["hq", "branch"], size=n)
        y_pred = np.where(
            (gender == "M") & (site == "hq"), rng.random(n) < 0.9, rng.random(n) < 0.3
        ).astype(int)
        df = pd.DataFrame({"gender": gender, "site": site})
        real = intersectional_module.identify_privileged_groups

        def singles_fail(y_true, y_hat, attr, **kw):
            if set(np.asarray(attr).tolist()) <= {"F", "M", "hq", "branch"}:
                raise ValueError("simulated single-attribute failure")
            return real(y_true, y_hat, attr, **kw)

        monkeypatch.setattr(intersectional_module, "identify_privileged_groups", singles_fail)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = discover_intersectional_groups(
                df, ["gender", "site"], y_pred, min_group_size=30
            )

        assert len(results) == 1
        r = results[0]
        assert math.isfinite(r["disparity"]) and r["disparity"] > 0.05
        assert math.isnan(r["hidden_disparity"])
        assert r["reveals_more"] is None
        assert r["single_attributes_not_assessable"] == ["gender", "site"]
        assert _messages(w, "no single-attribute disparity was assessable for ['gender', 'site']")
        assert _messages(w, "reveals_more=None")

    def test_rank_fairness_issues_reports_an_uncompared_intersection_as_could_not_check(
        self, monkeypatch
    ):
        """The consumer used to read reveals_more with a truthiness test, so a
        None collapsed into 'no hidden disparity found'."""
        rng = np.random.default_rng(0)
        n = 400
        gender = rng.choice(["F", "M"], size=n)
        site = rng.choice(["hq", "branch"], size=n)
        y_pred = np.where(
            (gender == "M") & (site == "hq"), rng.random(n) < 0.9, rng.random(n) < 0.3
        ).astype(int)
        df = pd.DataFrame({"gender": gender, "site": site})
        real = intersectional_module.identify_privileged_groups

        def singles_fail(y_true, y_hat, attr, **kw):
            if set(np.asarray(attr).tolist()) <= {"F", "M", "hq", "branch"}:
                raise ValueError("simulated single-attribute failure")
            return real(y_true, y_hat, attr, **kw)

        monkeypatch.setattr(intersectional_module, "identify_privileged_groups", singles_fail)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analysis = rank_fairness_issues(
                df,
                y_pred,
                protected_columns=["gender", "site"],
                auto_detect=False,
                min_group_size=30,
            )

        assert analysis["intersectional_issues"][0]["reveals_more"] is None
        could_not = [
            rec
            for rec in analysis["recommendations"]
            if rec.startswith("COULD NOT CHECK") and "could not be compared" in rec
        ]
        assert could_not, analysis["recommendations"]
        assert not any(rec.startswith("INTERSECTIONAL:") for rec in analysis["recommendations"])

    def test_control_two_measured_attributes_report_the_real_hidden_disparity(self):
        """Over-correction control: with both attributes assessable the result
        is a number and a real True, with no warning from discovery."""
        rng = np.random.default_rng(0)
        n = 400
        gender = rng.choice(["F", "M"], size=n)
        race = rng.choice(["a", "b"], size=n)
        y_pred = np.where(
            (gender == "M") & (race == "a"), rng.random(n) < 0.9, rng.random(n) < 0.3
        ).astype(int)
        df = pd.DataFrame({"gender": gender, "race": race})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = discover_intersectional_groups(
                df, ["gender", "race"], y_pred, min_group_size=30
            )

        assert len(results) == 1
        r = results[0]
        assert r["single_attributes_not_assessable"] == []
        assert math.isfinite(r["hidden_disparity"]) and r["hidden_disparity"] > 0.1
        assert r["reveals_more"] is True
        assert not _messages(w, "discover_intersectional_groups")

    def test_control_a_real_zero_hidden_disparity_is_still_false_not_none(self):
        """Predictions are a deterministic function of 'gender' (M -> 1, F -> 0),
        so every gender-by-site cell has rate exactly 0 or 1: the intersection
        splits exactly like 'gender' alone, hidden is a measured 0 and
        reveals_more a measured False (a random draw would add sampling noise
        and make the intersection legitimately exceed the single split)."""
        rng = np.random.default_rng(0)
        n = 400
        gender = rng.choice(["F", "M"], size=n)
        site = rng.choice(["hq", "branch"], size=n)
        y_pred = (gender == "M").astype(int)
        df = pd.DataFrame({"gender": gender, "site": site})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = discover_intersectional_groups(
                df, ["gender", "site"], y_pred, min_group_size=30
            )

        assert len(results) == 1
        r = results[0]
        assert r["single_attributes_not_assessable"] == []
        assert r["reveals_more"] is False
        assert r["hidden_disparity"] == 0
        assert not _messages(w, "discover_intersectional_groups")


# ---------------------------------------------------------------------------
# R-1: adaptive_sampling_plan, budget exhausted is not "already powered"
# ---------------------------------------------------------------------------


def _analyzer(n_per_arm: int, attrs, seed: int = 0) -> FairnessPowerAnalyzer:
    rng = np.random.default_rng(seed)

    def arm(n):
        return pd.DataFrame(
            {
                "gender": rng.choice(["F", "M"], n),
                "age": rng.choice(["young", "old"], n),
                "approved": rng.integers(0, 2, n),
            }
        )

    experiment = FairnessExperiment(arm(n_per_arm), arm(n_per_arm), list(attrs), "approved")
    return FairnessPowerAnalyzer(experiment)


class TestR1BudgetExhaustedIsNotAlreadyPowered:
    """Pre-fix, measured 2026-09-09: four intersections all at power 0.00
    needing 393 per arm, budget=2. The plan reported two of them as
    additional_samples=0 with rationale "already powered", while its own
    n_underpowered said 4, and it warned nothing."""

    def test_unreached_intersections_are_not_allocated_and_say_the_budget_ran_out(self):
        analyzer = _analyzer(40, ["gender", "age"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=2, effect_size=0.2)

        assert plan.n_underpowered == 4
        assert len(plan.allocations) == 2
        assert len(plan.not_allocated) == 2
        assert set(plan.not_allocated).isdisjoint(plan.allocations)
        for ix in plan.not_allocated:
            assert "NOT ALLOCATED" in plan.rationale[ix]
            assert "budget of 2" in plan.rationale[ix]
            assert "exhausted" in plan.rationale[ix]
        assert "already powered" not in plan.rationale.values()

        df = plan.to_dataframe()
        assert len(df) == plan.n_underpowered
        by_ix = dict(zip(df["intersection"], df["additional_samples"]))
        for ix in plan.not_allocated:
            assert pd.isna(by_ix[str(ix)]), by_ix
        for ix, alloc in plan.allocations.items():
            assert by_ix[str(ix)] == alloc
        assert (df["additional_samples"] == 0).sum() == 0

        as_dict = plan.to_dict()
        assert as_dict["n_not_allocated"] == 2
        assert len(as_dict["not_allocated"]) == 2

        named = _messages(w, "adaptive_sampling_plan: the budget of 2 was exhausted")
        assert named and "2 of 4 underpowered" in named[0]
        assert "additional_samples is None, not 0" in named[0]

    def test_zero_budget_allocates_nothing_and_still_counts_every_underpowered_one(self):
        analyzer = _analyzer(40, ["gender", "age"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=0, effect_size=0.2)

        assert plan.allocations == {}
        assert plan.total_budget == 0
        assert plan.n_underpowered == 4
        assert len(plan.not_allocated) == 4
        assert "already powered" not in plan.rationale.values()
        assert plan.to_dataframe()["additional_samples"].isna().all()
        assert _messages(w, "4 of 4 underpowered")

    def test_control_a_budget_that_reaches_everyone_is_unchanged(self):
        """Over-correction control: no NaN, no not_allocated, no warning, and
        the whole budget is spent."""
        analyzer = _analyzer(40, ["gender", "age"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=400, effect_size=0.2)

        assert plan.not_allocated == []
        assert plan.n_underpowered == 4 == len(plan.allocations)
        assert sum(plan.allocations.values()) == 400 == plan.total_budget
        df = plan.to_dataframe()
        assert not df["additional_samples"].isna().any()
        assert (df["additional_samples"] > 0).all()
        assert "already powered" not in plan.rationale.values()
        assert not _messages(w, "adaptive_sampling_plan")

    def test_control_a_fully_powered_experiment_still_reads_already_powered(self):
        analyzer = _analyzer(2000, ["gender"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = analyzer.adaptive_sampling_plan(budget=100, effect_size=0.2)

        assert plan.n_underpowered == 0
        assert plan.allocations == {}
        assert plan.not_allocated == []
        assert set(plan.rationale.values()) == {"already powered"}
        assert not _messages(w, "adaptive_sampling_plan")


# ---------------------------------------------------------------------------
# PARETO: compute_pareto_frontier, a variant missing a metric cannot be ranked
# ---------------------------------------------------------------------------


class TestParetoFrontierDoesNotRankAnUnmeasuredVariant:
    """Pre-fix, measured 2026-09-09: A={acc:0.5,fair:0.9}, B={acc:0.9} gave
    both is_pareto_optimal=True with no warning; B's fairness was unknown and
    entered the comparison as 0. And because the comparison set was the
    FIRST variant's keys, B listed first dropped 'fair' from the comparison
    entirely."""

    def _analysis(self) -> ExperimentAnalysis:
        return ExperimentAnalysis(experiment_result=None)

    def test_variant_missing_a_metric_gets_none_and_is_named(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            points = self._analysis().compute_pareto_frontier(
                {"A": {"acc": 0.5, "fair": 0.9}, "B": {"acc": 0.9}}
            )

        by_name = {p.variant: p for p in points}
        assert by_name["B"].is_pareto_optimal is None
        assert by_name["B"].missing_metrics == ["fair"]
        assert by_name["B"].dominates == []
        assert by_name["A"].is_pareto_optimal is True
        assert by_name["A"].missing_metrics == []
        assert by_name["B"].to_dict()["is_pareto_optimal"] is None
        assert by_name["B"].to_dict()["missing_metrics"] == ["fair"]
        named = _messages(
            w, "compute_pareto_frontier: variant 'B' has no measured value for ['fair']"
        )
        assert named and "is_pareto_optimal=None" in named[0]

    def test_the_comparison_set_is_the_union_not_the_first_variants_keys(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            points = self._analysis().compute_pareto_frontier(
                {"B": {"acc": 0.9}, "A": {"acc": 0.5, "fair": 0.9}}
            )

        by_name = {p.variant: p for p in points}
        assert by_name["B"].is_pareto_optimal is None
        assert by_name["B"].missing_metrics == ["fair"]
        assert by_name["A"].is_pareto_optimal is True
        assert _messages(w, "variant 'B'")

    @pytest.mark.parametrize("unmeasured", [float("nan"), None])
    def test_nan_and_none_count_as_unmeasured(self, unmeasured):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            points = self._analysis().compute_pareto_frontier(
                {"A": {"acc": 0.5, "fair": 0.9}, "B": {"acc": 0.9, "fair": unmeasured}}
            )

        by_name = {p.variant: p for p in points}
        assert by_name["B"].is_pareto_optimal is None
        assert by_name["B"].missing_metrics == ["fair"]
        assert by_name["A"].is_pareto_optimal is True
        assert _messages(w, "variant 'B'")
        # to_dict must survive the unmeasured value instead of rounding it
        assert "fair" in by_name["B"].to_dict()["metrics"]

    def test_control_complete_variants_rank_exactly_as_before(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            points = self._analysis().compute_pareto_frontier(
                {
                    "treatment": {"revenue": 1.05, "fairness": 0.95},
                    "control": {"revenue": 1.00, "fairness": 0.98},
                    "worse": {"revenue": 0.90, "fairness": 0.90},
                }
            )

        verdicts = [
            (p.variant, p.is_pareto_optimal, p.dominates, p.missing_metrics) for p in points
        ]
        assert verdicts == [
            ("treatment", True, ["worse"], []),
            ("control", True, ["worse"], []),
            ("worse", False, [], []),
        ]
        assert not _messages(w, "compute_pareto_frontier")

    def test_control_minimised_metrics_still_flip_sign(self):
        points = self._analysis().compute_pareto_frontier(
            {"x": {"cost": 1.0, "gain": 2.0}, "y": {"cost": 2.0, "gain": 2.0}},
            maximize=["gain"],
        )
        assert [(p.variant, p.is_pareto_optimal, p.dominates) for p in points] == [
            ("x", True, ["y"]),
            ("y", False, []),
        ]

    def test_control_empty_input_is_empty(self):
        assert self._analysis().compute_pareto_frontier({}) == []


# ---------------------------------------------------------------------------
# F18: compute_metric_with_ci, 'bayesian' is documented as not implemented
# ---------------------------------------------------------------------------


def _two_groups(per_group: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    n = 2 * per_group
    g = np.array(["a"] * per_group + ["b"] * per_group)
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(g == "a", rng.random(n) < 0.7, rng.random(n) < 0.4).astype(int)
    return y_true, y_pred, g


class TestF18BayesianIsDocumentedAsNotImplemented:
    """Pre-fix, measured 2026-09-09: method='auto' with two groups of 20
    warned "Bayesian method requested but using bootstrap ..." for a request
    nobody made, because 'auto' selected a 'bayesian' branch that has no
    estimator behind it. result.method was truthful throughout
    ('stratified_bootstrap_fold_debiased'); the documented option was not."""

    def test_requesting_bayesian_warns_and_reports_the_bootstrap_that_ran(self):
        y_true, y_pred, g = _two_groups(20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                g,
                n_bootstrap=200,
                method="bayesian",
                random_state=0,
                min_group_size=1,
            )

        named = _messages(w, "Bayesian method requested but using bootstrap")
        assert named and "not implemented" in named[0]
        assert result.method.startswith("stratified_bootstrap")
        assert "bayes" not in result.method.lower()
        assert result.n_bootstrap == 400

    def test_auto_on_small_groups_does_not_claim_a_bayesian_request(self):
        y_true, y_pred, g = _two_groups(20)
        assert 20 < SMALL_SAMPLE_THRESHOLD
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                g,
                n_bootstrap=200,
                method="auto",
                random_state=0,
                min_group_size=1,
            )

        assert not _messages(w, "Bayesian method requested")
        small = _messages(w, "Small sample (min group: 20")
        assert small and "A Bayesian interval is not implemented" in small[0]
        assert result.method.startswith("stratified_bootstrap")
        assert result.n_bootstrap == 400

    def test_auto_no_longer_names_a_branch_that_does_not_exist(self):
        source = inspect.getsource(compute_metric_with_ci)
        assert 'method = "bayesian"' not in source
        # Both places the docstring describes 'bayesian' must say it is not
        # implemented: the method-selection paragraph and the Args entry. A
        # looser `"NOT implemented" in doc` stayed green when one of the two
        # was sabotaged (sabotage check, 2026-09-09).
        doc = " ".join(compute_metric_with_ci.__doc__.split())
        assert (
            "``method='bayesian'`` is accepted for API compatibility but is NOT implemented" in doc
        )
        assert "'bayesian' (NOT implemented: warns and falls back" in doc

    def test_unknown_method_is_refused_not_silently_bootstrapped(self):
        y_true, y_pred, g = _two_groups(60)
        with pytest.raises(ConfigurationError, match="method must be"):
            compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                g,
                n_bootstrap=50,
                method="mixed",
                min_group_size=1,
            )

    def test_control_auto_on_adequate_groups_is_a_plain_bootstrap_without_warning(self):
        y_true, y_pred, g = _two_groups(60)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_metric_with_ci(
                demographic_parity_difference,
                y_true,
                y_pred,
                g,
                n_bootstrap=200,
                method="auto",
                random_state=0,
                min_group_size=1,
            )

        assert not _messages(w, "Bayesian")
        assert not _messages(w, "Small sample")
        assert result.method.startswith("stratified_bootstrap")
        assert result.n_bootstrap == 200
        assert math.isfinite(result.lower_bound) and math.isfinite(result.upper_bound)
