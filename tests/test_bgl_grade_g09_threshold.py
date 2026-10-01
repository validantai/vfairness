"""BGL grade wave, batch G09:
vfairness.post_processing.threshold_optimization (analyzer + constraints +
optimizer containers).

THESE UNITS CARRY A MITIGATION'S VERDICT, so the failure mode is inverted: a
constraint that has stopped constraining reports a BETTER fairness number, not
a worse one. The pins below therefore read the SEARCH's and the CONSTRAINT's own
parameters, never the fairness number they publish.

The defect this file was written for (2026-09-30): ThresholdConstraint accepted
a tolerance OUTSIDE the metric's range, which makes `violation <= tolerance` a
comparison that cannot be false for any data. Measured on 200 rows in two
groups where group 'a' is accepted at every threshold and 'b' at none, so the
demographic-parity disparity is 1.0000, the largest one can be:

    GroupThresholdOptimizer(tolerance=0.05) -> is_feasible False    correct
    GroupThresholdOptimizer(tolerance=2.0)  -> is_feasible True,
        summary() "Feasible: True", violation 1.0000, NO could-not-check
        block and ZERO warnings

Two more doors were open beside it: FeasibleThresholds.get(group, []) handed
back the empty list that the class exists to refuse (dict.get does not call
__missing__), and BaseThresholdOptimizer stored any non-constraint object as
self.constraint unchecked.
"""

import json
import math

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    BoundRole,
    metric_value_range,
    vacuous_bound_reason,
)
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalysisReport,
    ThresholdImpactResult,
)
from vfairness.post_processing.threshold_optimization.constraints import (
    ConstraintViolation,
    FairnessConstraintType,
    FeasibleThresholds,
)
from vfairness.post_processing.threshold_optimization.optimizer import (
    BaseThresholdOptimizer,
    GroupThresholdOptimizer,
    ThresholdConstraint,
    ThresholdResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures: a MAXIMAL disparity, so a vacuous bound is unmistakable
# ---------------------------------------------------------------------------


def _maximal_disparity_batch():
    """200 rows where group 'a' scores high and 'b' low: the gap is 1.0000."""
    rng = np.random.default_rng(0)
    sensitive = np.array(["a"] * 100 + ["b"] * 100)
    y_prob = np.concatenate([rng.uniform(0.90, 0.99, 100), rng.uniform(0.01, 0.10, 100)])
    y_true = np.concatenate([np.ones(100, int), np.zeros(100, int)])
    return y_true, y_prob, sensitive


def _violation(**kw):
    base = dict(
        constraint_type=FairnessConstraintType.DEMOGRAPHIC_PARITY,
        violation=0.01,
        group_metrics={"a": 0.50, "b": 0.49},
        group_violations={"a": 0.005, "b": 0.005},
        is_satisfied=True,
        tolerance=0.05,
    )
    base.update(kw)
    return ConstraintViolation(**base)


# ---------------------------------------------------------------------------
# ThresholdConstraint: the bound that decides every verdict
# ---------------------------------------------------------------------------


# Derived from the metric's declared range, not from a quoted list of numbers:
# whatever the range becomes, the vacuous bound is computed from it here.
_RANGED = [c for c in FairnessConstraintType if metric_value_range(c.value) is not None]


@pytest.mark.parametrize("constraint", _RANGED, ids=[c.value for c in _RANGED])
def test_a_tolerance_the_disparity_cannot_exceed_is_refused(constraint):
    low, high = metric_value_range(constraint.value)
    worst = max(abs(low), abs(high))
    for tolerance in (worst, worst + 0.2, worst * 100):
        # The repository's own rule agrees this bound grades nothing...
        assert vacuous_bound_reason(constraint.value, tolerance, BoundRole.THRESHOLD) is not None
        # ...so the constraint must refuse it.
        with pytest.raises(ValueError, match="cannot grade"):
            ThresholdConstraint(constraint_type=constraint, tolerance=tolerance)
        with pytest.raises(ValueError, match="cannot grade"):
            ThresholdConstraint.from_string(constraint.value, tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf"), -0.1, -1e-9])
def test_an_unreadable_or_negative_tolerance_is_refused_at_construction(tolerance):
    with pytest.raises(ValueError, match="finite, non-negative"):
        ThresholdConstraint(FairnessConstraintType.DEMOGRAPHIC_PARITY, tolerance=tolerance)


@pytest.mark.parametrize("weight", [0.0, -1.0, 2.0, 0.5])
def test_the_unimplemented_weight_is_refused_rather_than_ignored(weight):
    """Nothing reads ThresholdConstraint.weight. A knob that silently does
    nothing states a capability the code does not have, so it is refused, the
    disposition soft_rate_computation's `temperature` already received."""
    with pytest.raises(ValueError, match="weight is not implemented"):
        ThresholdConstraint(FairnessConstraintType.DEMOGRAPHIC_PARITY, weight=weight)


def test_a_usable_tolerance_is_still_accepted_with_its_real_value():
    """The control. A guard that refused every tolerance would pass every test
    above and make the package unusable."""
    for tolerance in (0.0, 1e-6, 0.05, 0.5, 0.999):
        built = ThresholdConstraint(FairnessConstraintType.DEMOGRAPHIC_PARITY, tolerance=tolerance)
        assert built.tolerance == tolerance
        assert built.weight == 1.0
    assert ThresholdConstraint.from_string("equalized_odds", 0.1) == ThresholdConstraint(
        FairnessConstraintType.EQUALIZED_ODDS, 0.1
    )
    assert ThresholdConstraint(FairnessConstraintType.DEMOGRAPHIC_PARITY).tolerance == 0.05
    for bad in ("not_a_constraint", "", None):
        with pytest.raises(ValueError, match="not a valid FairnessConstraintType"):
            ThresholdConstraint.from_string(bad)


def test_a_constraint_type_with_no_declared_range_keeps_its_wide_tolerance():
    """Deliberate non-refusal. vacuous_bound_reason answers None when the
    metric has no declared range, because whether the bound is reachable is
    then unknown, and refusing an unknown is the over-correction."""
    unranged = [c for c in FairnessConstraintType if metric_value_range(c.value) is None]
    assert unranged, "every constraint type now has a range; this test needs rewriting"
    for constraint in unranged:
        assert ThresholdConstraint(constraint_type=constraint, tolerance=5.0).tolerance == 5.0


def test_the_vacuous_tolerance_reached_the_optimizer_and_published_a_pass():
    """The end-to-end reproduction, and the control beside it."""
    y_true, y_prob, sensitive = _maximal_disparity_batch()
    # The disparity really is maximal: every 'a' row scores above every 'b' row.
    assert y_prob[:100].min() > y_prob[100:].max()

    honest = GroupThresholdOptimizer(
        constraint="demographic_parity", tolerance=0.05, n_thresholds=20
    )
    honest.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive)
    assert honest.result_.is_feasible is False
    assert "Feasible: False" in honest.result_.summary()

    # The bound that used to grade this data as satisfied cannot be built now.
    with pytest.raises(ValueError, match="cannot grade"):
        GroupThresholdOptimizer(constraint="demographic_parity", tolerance=2.0, n_thresholds=20)
    with pytest.raises(ValueError, match="cannot grade"):
        GroupThresholdOptimizer(constraint="demographic_parity", tolerance=1.0, n_thresholds=20)


# ---------------------------------------------------------------------------
# BaseThresholdOptimizer
# ---------------------------------------------------------------------------


class _Concrete(BaseThresholdOptimizer):
    """The smallest thing that satisfies the abstract interface."""

    def fit(self, *, y_true, y_prob, sensitive_attr, sample_weight=None):
        self.is_fitted = True
        self.result_ = ThresholdResult(0.5, {}, [], {"accuracy": 1.0}, True)
        return self

    def predict(self, y_prob, sensitive_attr):
        return (np.asarray(y_prob) >= 0.5).astype(int)


def test_the_abstract_optimizer_cannot_be_instantiated():
    assert BaseThresholdOptimizer.__abstractmethods__ == frozenset({"fit", "predict"})
    with pytest.raises(TypeError, match="abstract"):
        BaseThresholdOptimizer()


@pytest.mark.parametrize("bad", [123, None, 4.5, ["demographic_parity"], object()])
def test_a_constraint_that_is_not_a_constraint_is_refused_at_construction(bad):
    """The else branch was a bare assignment, so self.constraint became the
    object as given and the first read of constraint.tolerance inside fit()
    raised an AttributeError naming int or NoneType, not the argument."""
    with pytest.raises(TypeError, match="constraint must be"):
        _Concrete(constraint=bad)


def test_the_three_documented_constraint_spellings_all_work():
    """The control, one per accepted input type."""
    from_str = _Concrete(constraint="demographic_parity", tolerance=0.07)
    assert from_str.constraint == ThresholdConstraint(
        FairnessConstraintType.DEMOGRAPHIC_PARITY, 0.07
    )
    from_enum = _Concrete(constraint=FairnessConstraintType.EQUALIZED_ODDS, tolerance=0.07)
    assert from_enum.constraint.constraint_type is FairnessConstraintType.EQUALIZED_ODDS
    assert from_enum.constraint.tolerance == 0.07

    supplied = ThresholdConstraint(FairnessConstraintType.EQUAL_OPPORTUNITY, 0.2)
    passed_through = _Concrete(constraint=supplied, tolerance=0.99)
    # A supplied constraint keeps ITS OWN tolerance; the argument does not
    # silently overwrite it.
    assert passed_through.constraint is supplied
    assert passed_through.constraint.tolerance == 0.2

    # A bad name still fails on the name, not on the type.
    with pytest.raises(ValueError, match="not a valid FairnessConstraintType"):
        _Concrete(constraint="nope")


def test_the_unfitted_state_is_an_absence_not_a_result():
    fresh = _Concrete()
    assert fresh.is_fitted is False
    assert fresh.result_ is None  # not an empty ThresholdResult


def test_fit_predict_runs_both_halves_in_order():
    optimizer = _Concrete()
    out = optimizer.fit_predict(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]), np.array(["a", "a", "b", "b"])
    )
    assert out.tolist() == [1, 0, 1, 0]
    assert optimizer.is_fitted is True and optimizer.result_ is not None


# ---------------------------------------------------------------------------
# FeasibleThresholds: the could-not-check that .get walked around
# ---------------------------------------------------------------------------


def _feasible_thresholds():
    return FeasibleThresholds(
        {"a": [0.1, 0.2]},
        not_assessable={"b": "b has no negative labels, so FPR is 0/0"},
        unmeasurable_thresholds={"a": 3},
    )


def test_a_group_that_was_not_assessed_cannot_be_read_as_an_empty_set():
    """dict.get does NOT call __missing__, so .get(group, []) returned the
    empty list the class docstring says is a MEASUREMENT ('no threshold
    works'), which is the exact fabrication this class exists to remove."""
    thresholds = _feasible_thresholds()
    for read in (
        lambda: thresholds["b"],
        lambda: thresholds.get("b"),
        lambda: thresholds.get("b", []),
        lambda: thresholds.get("b", [0.5]),
        lambda: thresholds.setdefault("b", []),
    ):
        with pytest.raises(KeyError, match="NOT assessed"):
            read()
    # setdefault did not insert anything on its way out.
    assert dict(thresholds) == {"a": [0.1, 0.2]}
    # The reason is machine readable, not only in the message.
    assert "b" in thresholds.not_assessable
    assert thresholds.all_groups_assessed is False


def test_the_ordinary_dict_behaviour_is_untouched():
    """The control: the refusal is scoped to not_assessable and nothing else."""
    thresholds = _feasible_thresholds()
    assert thresholds["a"] == [0.1, 0.2]
    assert thresholds.get("a") == [0.1, 0.2]
    assert thresholds.get("zz") is None
    assert thresholds.get("zz", []) == []
    with pytest.raises(KeyError):
        thresholds["zz"]
    assert thresholds.setdefault("c", [0.5]) == [0.5]
    assert dict(thresholds) == {"a": [0.1, 0.2], "c": [0.5]}
    assert list(thresholds) == ["a", "c"] and len(thresholds) == 2
    assert thresholds.unmeasurable_thresholds == {"a": 3}
    assert "b has no negative labels" in repr(thresholds)

    empty = FeasibleThresholds()
    assert dict(empty) == {} and empty.all_groups_assessed is True
    # Per-instance mutable state, not shared class state.
    first, second = FeasibleThresholds(), FeasibleThresholds()
    first.not_assessable["x"] = "why"
    first.unmeasurable_thresholds["x"] = 1
    assert second.not_assessable == {} and second.unmeasurable_thresholds == {}


# ---------------------------------------------------------------------------
# ConstraintViolation and its dict: three states at the boundary
# ---------------------------------------------------------------------------


def test_constraint_violation_carries_all_three_states_and_serialises_them():
    measured = _violation(violation=0.12, is_satisfied=False)
    assert measured.to_dict()["is_satisfied"] is False
    assert measured.to_dict()["unmeasured"] == []
    assert measured.to_dict()["violation_is_lower_bound"] is False

    could_not_check = _violation(
        violation=float("nan"), is_satisfied=None, unmeasured=("b.fpr",), group_violations={}
    )
    out = could_not_check.to_dict()
    # None must NOT be serialised as False: that would be a measured failure
    # nobody measured.
    assert out["is_satisfied"] is None
    assert math.isnan(out["violation"])
    assert out["unmeasured"] == ["b.fpr"]

    lower_bound = _violation(
        violation=0.60, is_satisfied=False, unmeasured=("b.fpr",), violation_is_lower_bound=True
    )
    assert lower_bound.to_dict()["violation_is_lower_bound"] is True
    assert lower_bound.to_dict()["unmeasured"] == ["b.fpr"]

    # Every field reaches the dict, and the enum is serialised by VALUE so a
    # non-Python consumer can read it.
    assert set(ConstraintViolation.__dataclass_fields__) == set(measured.to_dict())
    assert measured.to_dict()["constraint_type"] == "demographic_parity"
    # unmeasured is a LIST in the dict (a tuple is not JSON), and a copy.
    assert json.loads(json.dumps(lower_bound.to_dict()))["unmeasured"] == ["b.fpr"]
    assert lower_bound.to_dict()["unmeasured"] is not lower_bound.unmeasured


def test_constraint_violation_defaults_do_not_claim_completeness_falsely():
    """unmeasured=() and violation_is_lower_bound=False are the right defaults
    ONLY because they describe a fully measured violation; assert they are not
    silently applied to a NaN one."""
    bare = _violation()
    assert bare.unmeasured == () and bare.violation_is_lower_bound is False
    first, second = _violation(), _violation()
    first.group_metrics["c"] = 0.1
    assert "c" not in second.group_metrics


# ---------------------------------------------------------------------------
# FeasibleRegion: two endpoints cannot carry three states
# ---------------------------------------------------------------------------


def test_feasible_region_separates_infeasible_from_not_assessed():
    feasible = FeasibleRegion(0.2, 0.8, assessed=True, n_searched=100, n_feasible=60)
    infeasible = FeasibleRegion(None, None, assessed=True, n_searched=100, n_feasible=0)
    unassessed = FeasibleRegion(None, None, assessed=False, n_searched=100, n_not_assessed=100)

    assert (feasible.status, infeasible.status, unassessed.status) == (
        "feasible",
        "infeasible",
        "not_assessed",
    )
    # The two (None, None) cases are indistinguishable as PAIRS, which is the
    # whole reason the class exists.
    assert tuple(infeasible) == tuple(unassessed) == (None, None)
    assert infeasible.status != unassessed.status


def test_feasible_region_is_still_a_two_element_tuple_for_every_old_caller():
    region = FeasibleRegion(0.2, 0.8)
    assert isinstance(region, tuple) and len(region) == 2
    low, high = region
    assert (low, high) == (0.2, 0.8) and region[0] == 0.2 and region[1] == 0.8
    assert FeasibleRegion(None, None) == (None, None)
    assert hash(FeasibleRegion(0.2, 0.8)) == hash((0.2, 0.8))


def test_the_hand_built_region_does_not_default_to_a_reassuring_answer():
    """score_resolution_sufficient must be None, not True: a reassuring default
    for an absent measurement is the shape this class exists to remove."""
    region = FeasibleRegion(None, None)
    assert region.n_distinct_scores is None
    assert region.n_distinct_decisions is None
    assert region.score_resolution_sufficient is None
    assert region.n_searched == 0 and region.n_feasible == 0


# ---------------------------------------------------------------------------
# ThresholdImpactResult / ThresholdAnalysisReport / ThresholdResult
# ---------------------------------------------------------------------------


def _impact_result(**kw):
    base = dict(
        threshold=0.5,
        performance_metrics={"accuracy": 0.80},
        group_metrics={"a": {"tpr": 0.90}},
        constraint_violations={
            "equalized_odds": _violation(
                constraint_type=FairnessConstraintType.EQUALIZED_ODDS,
                violation=float("nan"),
                is_satisfied=None,
                unmeasured=("b.fpr",),
                group_violations={},
            )
        },
        confusion_matrix={"tp": 10, "fp": 2, "tn": 20, "fn": 3},
        group_confusion_matrices={"a": {"tp": 5}},
    )
    base.update(kw)
    return ThresholdImpactResult(**base)


def test_threshold_impact_result_is_a_container_that_invents_nothing():
    result = _impact_result()
    assert result.threshold == 0.5
    assert result.performance_metrics == {"accuracy": 0.80}
    nan_result = _impact_result(
        threshold=float("nan"), performance_metrics={"accuracy": float("nan")}
    )
    assert math.isnan(nan_result.threshold)
    assert math.isnan(nan_result.performance_metrics["accuracy"])


def test_threshold_impact_result_to_dict_keeps_the_nested_third_state():
    out = _impact_result().to_dict()
    assert set(ThresholdImpactResult.__dataclass_fields__) == set(out)
    nested = out["constraint_violations"]["equalized_odds"]
    # The nested violation is SERIALISED, not handed over as an object, and its
    # could-not-check survives the boundary.
    assert isinstance(nested, dict)
    assert nested["is_satisfied"] is None
    assert nested["unmeasured"] == ["b.fpr"]
    assert nested["constraint_type"] == "equalized_odds"
    # Control: a measured one keeps its real number.
    measured = _impact_result(
        constraint_violations={"demographic_parity": _violation(violation=0.12, is_satisfied=False)}
    ).to_dict()
    assert measured["constraint_violations"]["demographic_parity"]["violation"] == 0.12
    assert measured["constraint_violations"]["demographic_parity"]["is_satisfied"] is False


def _report():
    return ThresholdAnalysisReport(
        threshold_results=[_impact_result()],
        optimal_thresholds={"demographic_parity": {"accuracy": None, "f1_score": 0.42}},
        feasible_regions={
            "feasible": FeasibleRegion(
                0.2, 0.8, assessed=True, n_searched=100, n_feasible=60, n_distinct_decisions=42
            ),
            "gapped": FeasibleRegion(
                0.01,
                0.99,
                assessed=True,
                n_searched=100,
                n_feasible=12,
                contiguous=False,
                n_distinct_decisions=12,
            ),
            "infeasible": FeasibleRegion(
                None, None, assessed=True, n_searched=100, n_feasible=0, n_distinct_decisions=42
            ),
            "not_assessed": FeasibleRegion(
                None, None, assessed=False, n_searched=100, n_not_assessed=100
            ),
            "one_decision": FeasibleRegion(
                0.01,
                0.99,
                assessed=True,
                n_searched=100,
                n_feasible=100,
                n_distinct_decisions=1,
                n_distinct_scores=200,
                score_resolution_sufficient=True,
            ),
            "bare_tuple": (0.3, 0.7),
        },
        recommendations=["use group-specific thresholds"],
    )


def test_the_report_dict_distinguishes_every_region_state_for_a_json_consumer():
    regions = _report().to_dict()["feasible_regions"]
    assert regions["infeasible"]["status"] == "infeasible"
    assert regions["not_assessed"]["status"] == "not_assessed"
    # The two used to encode identically as [null, null].
    assert (regions["infeasible"]["lower"], regions["infeasible"]["upper"]) == (None, None)
    assert (regions["not_assessed"]["lower"], regions["not_assessed"]["upper"]) == (None, None)
    assert regions["infeasible"] != regions["not_assessed"]
    # A bare pair claims nothing rather than claiming it was assessed.
    assert regions["bare_tuple"]["status"] == "unknown"
    assert "assessed" not in regions["bare_tuple"]
    # The "100 searched thresholds, 1 decision" case is legible here too.
    assert regions["one_decision"]["n_distinct_decisions"] == 1
    assert regions["one_decision"]["n_searched"] == 100
    assert regions["gapped"]["contiguous"] is False
    # Every field reaches the dict, and it really encodes.
    assert set(ThresholdAnalysisReport.__dataclass_fields__) == set(_report().to_dict())
    json.dumps(_report().to_dict(), default=str)
    # The ATTRIBUTE keeps the pair interface for existing callers.
    assert _report().feasible_regions["not_assessed"] == (None, None)


def test_the_report_summary_says_which_of_the_three_states_each_region_is_in():
    text = _report().summary()
    assert "feasible: [0.200, 0.800]" in text
    assert "NOT ASSESSED (no threshold could be checked" in text
    assert "not a finding of infeasibility" in text
    assert "No feasible region found" in text
    # The gap is INSIDE the brackets the reader acts on, not only in a flag.
    assert "NOT as one run" in text
    # And the one-decision caveat is attached to the line it qualifies.
    one_decision_line = [line for line in text.splitlines() if "one_decision" in line][0]
    assert "NOT A SEARCH" in one_decision_line
    assert "trivially equal across groups" in one_decision_line
    # An unmeasurable optimum is named, not printed as 0.000.
    assert "Best for accuracy: NOT ASSESSED" in text
    assert "Best for f1_score: 0.420" in text


def test_an_empty_report_reports_emptiness_rather_than_zeroes():
    empty = ThresholdAnalysisReport([], {}, {}, [])
    assert empty.to_dict() == {
        "threshold_results": [],
        "optimal_thresholds": {},
        "feasible_regions": {},
        "recommendations": [],
        "metadata": {},
    }
    text = empty.summary()
    assert "Analyzed 0 threshold values" in text
    assert "Recommendations" not in text  # nothing invented to fill the section


def test_threshold_result_prints_the_three_states_and_every_caveat():
    measured = ThresholdResult(0.42, {}, [_violation()], {"accuracy": 0.80}, True)
    assert "Feasible: True" in measured.summary()
    assert "✓ demographic_parity: 0.0100" in measured.summary()
    assert measured.to_dict()["is_feasible"] is True

    unassessed = ThresholdResult(
        None,
        {"a": 0.3, "b": 0.6},
        [
            _violation(
                constraint_type=FairnessConstraintType.EQUALIZED_ODDS,
                violation=float("nan"),
                is_satisfied=None,
                unmeasured=("b.fpr",),
            )
        ],
        {"accuracy": 0.80},
        None,
        {"constraint_measured": False, "n_violation_candidates": 50},
    )
    text = unassessed.summary()
    # None is never rendered as False.
    assert "Feasible: not assessed (constraint could not be measured)" in text
    assert "? equalized_odds" in text
    assert "COULD NOT CHECK:" in text
    assert "NOT a minimum" in text
    assert unassessed.to_dict()["is_feasible"] is None

    every_caveat = ThresholdResult(
        0.5,
        {},
        [
            _violation(
                constraint_type=FairnessConstraintType.EQUALIZED_ODDS,
                violation=0.60,
                is_satisfied=False,
                unmeasured=("b.fpr",),
                violation_is_lower_bound=True,
            )
        ],
        {"accuracy": 0.80},
        False,
        {
            "objective_measured": False,
            "objective": "f1_score",
            "score_resolution_sufficient": False,
            "n_distinct_scores": 2,
            "pareto_frontier_computed": False,
            "pareto_fallback_reason": "3 groups, so the sweep did not run",
            "objectives": ["accuracy", "f1_score"],
            "objectives_unmeasured": ["f1_score"],
            "objectives_measured": ["accuracy"],
            "n_unscored_rows": 10,
            "n_rows_fitted": 300,
        },
    )
    caveats = every_caveat.summary()
    assert "at least 0.6000 (lower bound" in caveats
    for expected in (
        "LOWER BOUND",
        "NOT an optimum",
        "NOT over the objectives that were requested",
        "NO Pareto frontier was computed",
        "no threshold sweep could separate anything",
        "entered the fairness measurement as REJECTED",
    ):
        assert expected in caveats, expected
    assert set(ThresholdResult.__dataclass_fields__) == set(every_caveat.to_dict())
    json.dumps(every_caveat.to_dict(), default=str)
