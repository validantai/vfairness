"""First examination of the 19 public units added for vfairness issue #1 (2026-10-24).

Beta condition B1 ("every public code unit has been executed and its result
recorded") stood at 1,580 of 1,580 before deployment policies and the structured
scope record were added. Each new unit is run here directly, on the input it
exists for and on a control, and every ``to_dict`` is pinned in full, because
serialisers were the most overturned dismissals (docs/GRADING.md). Each test was
sabotaged and seen to fail before the grade was recorded in the grading wave of
2026-10-24.

The behaviour these units carry is pinned more widely in
tests/test_issue1_coverage_policy.py; this file is the per-unit examination.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vfairness.exceptions import ConfigurationError
from vfairness.operations.cicd.coverage import (
    CoverageItem,
    CoveragePolicy,
    CoverageState,
    MeasuredOutcome,
    PolicyDecision,
    assess_coverage,
    policy_check_conclusion,
    policy_result_line,
)
from vfairness.operations.cicd.gate import (
    EvaluatedScope,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    ModelFairnessGate,
    ScopeGap,
    ScopeLevel,
)

DPD = "demographic_parity_difference"


def _cells():
    sex = np.array(["f"] * 200 + ["m"] * 200)
    race = np.array((["x"] * 100 + ["y"] * 100) * 2)
    rate = {("f", "x"): 0.8, ("f", "y"): 0.2, ("m", "x"): 0.2, ("m", "y"): 0.8}
    unfair = np.array(
        [1 if i % 10 < rate[(s, r)] * 10 else 0 for i, (s, r) in enumerate(zip(sex, race))]
    )
    fair = np.array([i % 2 for i in range(400)])
    return sex, race, unfair, fair


def _policy(**over):
    kw = dict(
        policy_id="p",
        version="1",
        required_attributes=["sex", "race"],
        required_intersections=[("race", "sex")],
        required_metrics={DPD: 0.1},
        min_group_size=30,
    )
    kw.update(over)
    return CoveragePolicy(**kw)


def _gate(threshold=0.1):
    return ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: threshold}))


def _decide(y, hconfig=None, policy=None, gate=None):
    sex, race, _, _ = _cells()
    return (gate or _gate()).evaluate_against_policy(
        policy or _policy(),
        y,
        y,
        {"sex": sex, "race": race},
        hierarchical_config=hconfig
        or HierarchicalGateConfig(default_intersection_threshold_multiplier=1.0),
    )


# --- CoverageState, MeasuredOutcome --------------------------------------------


def test_coverage_state_has_exactly_the_four_states_and_prints_them_in_words():
    assert [s.value for s in CoverageState] == [
        "assessed",
        "insufficient_evidence",
        "omitted_by_configuration",
        "input_absent",
    ]
    assert CoverageState.OMITTED_BY_CONFIGURATION.label == "omitted by configuration"
    assert CoverageState.ASSESSED.label == "assessed"


def test_measured_outcome_has_exactly_three_outcomes_and_prints_them_in_words():
    assert [o.value for o in MeasuredOutcome] == ["pass", "fail", "not_measured"]
    assert MeasuredOutcome.NOT_MEASURED.label == "not measured"


# --- CoveragePolicy ------------------------------------------------------------


def test_coverage_policy_normalises_a_healthy_policy_and_refuses_a_bare_string():
    p = _policy(required_intersections=[("race", "sex"), ["sex", "race"]])
    assert p.required_attributes == ("sex", "race")
    assert p.required_intersections == (("sex", "race"),)
    assert p.group_sets == (("sex",), ("race",), ("sex", "race"))
    # "race" is a string, and a string is a sequence of characters.
    with pytest.raises(ConfigurationError, match="list or tuple"):
        _policy(required_attributes="race")


def test_coverage_policy_to_dict_is_the_whole_requirement():
    assert _policy().to_dict() == {
        "policy_id": "p",
        "version": "1",
        "required_attributes": ["sex", "race"],
        "required_intersections": [["sex", "race"]],
        "required_metrics": {DPD: 0.1},
        "min_group_size": 30,
    }
    assert json.loads(json.dumps(_policy().to_dict())) == _policy().to_dict()


# --- CoverageItem --------------------------------------------------------------


def _item(state, outcome, **over):
    kw = dict(
        attributes=("sex", "race"),
        metric=DPD,
        state=state,
        outcome=outcome,
        reason="r",
        level="intersection:sex_x_race",
        value=0.6,
        threshold_used=0.1,
        policy_bound=0.1,
        undersized=(("f_x", 5),),
        note="n",
    )
    kw.update(over)
    return CoverageItem(**kw)


def test_coverage_item_is_satisfied_only_when_assessed_and_passing():
    for state in CoverageState:
        for outcome in MeasuredOutcome:
            assert _item(state, outcome).satisfied is (
                state is CoverageState.ASSESSED and outcome is MeasuredOutcome.PASS
            )
    assert _item(CoverageState.ASSESSED, MeasuredOutcome.PASS).label == "sex × race"


def test_coverage_item_to_dict_carries_every_field_and_both_verdict_halves():
    assert _item(CoverageState.INSUFFICIENT_EVIDENCE, MeasuredOutcome.PASS).to_dict() == {
        "attributes": ["sex", "race"],
        "metric": DPD,
        "state": "insufficient_evidence",
        "outcome": "pass",
        "satisfied": False,
        "reason": "r",
        "level": "intersection:sex_x_race",
        "value": 0.6,
        "threshold_used": 0.1,
        "policy_bound": 0.1,
        "undersized": [["f_x", 5]],
        "note": "n",
    }


# --- PolicyDecision ------------------------------------------------------------


def test_policy_decision_names_its_gaps_and_summarises_them():
    _, _, unfair, fair = _cells()
    off = _decide(unfair, HierarchicalGateConfig(check_intersections=False))
    assert [(g.label, g.state) for g in off.gaps] == [
        ("sex × race", CoverageState.OMITTED_BY_CONFIGURATION)
    ]
    assert off.summary == (
        "NOT PERMITTED under policy p v1: 2 of 3 required item(s) assessed and passing"
    )
    clean = _decide(fair)
    assert clean.gaps == []
    assert clean.summary.startswith("PERMITTED under policy p v1: 3 of 3")


def test_policy_decision_to_dict_keeps_the_measurement_and_the_requirement():
    _, _, _, fair = _cells()
    d = _decide(fair)
    out = d.to_dict()
    assert set(out) == {
        "policy",
        "permitted",
        "items",
        "reasons",
        "measurement",
        "policy_fingerprint",
        "configuration",
        "configuration_fingerprint",
        "timestamp",
    }
    assert out["policy"] == d.policy.to_dict()
    assert out["measurement"] == d.measurement.to_dict()
    assert out["reasons"] == [] and out["permitted"] is True
    assert [i["state"] for i in out["items"]] == ["assessed"] * 3
    assert json.loads(json.dumps(out, default=str))["policy_fingerprint"] == d.policy_fingerprint


def test_policy_decision_markdown_report_carries_the_policy_and_the_measurement():
    _, _, unfair, _ = _cells()
    d = _decide(unfair, HierarchicalGateConfig(check_intersections=False))
    md = d.to_markdown_report()
    assert md.startswith("# Fairness Policy Report")
    assert "| sex × race | demographic_parity_difference | omitted by configuration |" in md
    assert "# Hierarchical Fairness Gate Report" in md
    assert md.index("Deployment Policy") < md.index("Hierarchical Fairness Gate Report")


# --- assess_coverage, policy_result_line, policy_check_conclusion ---------------


def test_assess_coverage_re_judges_a_stored_measurement_against_a_newer_policy():
    sex, race, _, fair = _cells()
    y = fair.copy()
    y[:100] = 1  # f_x selected at 1.0: every attribute and the cell now carry a gap
    gate = _gate(threshold=0.5)
    hconfig = HierarchicalGateConfig(default_intersection_threshold_multiplier=1.0)
    stored = gate.evaluate_hierarchical(
        y, y, {"sex": sex, "race": race}, hierarchical_config=hconfig
    )
    before = json.dumps(stored.to_dict(), default=str, sort_keys=True)
    kwargs = dict(
        protected_attrs={"sex": sex, "race": race},
        y_true=y,
        y_pred=y,
        gate_config=gate.config,
        hierarchical_config=hconfig,
    )
    v1 = assess_coverage(_policy(required_metrics={DPD: 0.5}), stored, **kwargs)
    v2 = assess_coverage(_policy(version="2", required_metrics={DPD: 0.6}), stored, **kwargs)
    # The newer version accepts a looser bound, so the configured 0.5 is at least as strict.
    assert v1.permitted is True and v2.permitted is True
    v3 = assess_coverage(_policy(version="3", required_metrics={DPD: 0.2}), stored, **kwargs)
    # A stricter version: the configured 0.5 is looser, and the measured cell gap is a breach.
    assert v3.permitted is False
    cell = [i for i in v3.items if i.attributes == ("sex", "race")][0]
    assert cell.state is CoverageState.OMITTED_BY_CONFIGURATION
    assert cell.outcome is MeasuredOutcome.FAIL
    # Pure: the stored measurement is unchanged.
    assert json.dumps(stored.to_dict(), default=str, sort_keys=True) == before


def test_policy_result_line_says_permitted_or_not_with_the_policy_named():
    _, _, unfair, fair = _cells()
    assert policy_result_line(_decide(fair)) == "**Deployment policy**: `PERMITTED` under `p` v1"
    assert policy_result_line(
        _decide(unfair, HierarchicalGateConfig(check_intersections=False))
    ) == ("**Deployment policy**: `NOT PERMITTED` under `p` v1")


def test_policy_check_conclusion_is_failure_neutral_or_success():
    sex, race, unfair, fair = _cells()
    assert policy_check_conclusion(_decide(fair)) == "success"
    assert (
        policy_check_conclusion(_decide(unfair, HierarchicalGateConfig(check_intersections=False)))
        == "failure"
    )
    conditional = ModelFairnessGate(
        config=GateConfig(
            metrics=[DPD, "auroc_parity"],
            thresholds={DPD: 0.1, "auroc_parity": 0.1},
            blocking_metrics=[DPD],
        )
    )
    d = _decide(fair, gate=conditional)
    assert d.measurement.status is GateStatus.CONDITIONAL and d.permitted is True
    assert policy_check_conclusion(d) == "neutral"


# --- ScopeLevel, ScopeGap, EvaluatedScope ----------------------------------------


def _hierarchy(**h):
    sex, race, _, fair = _cells()
    return _gate().evaluate_hierarchical(
        fair, fair, {"sex": sex, "race": race}, hierarchical_config=HierarchicalGateConfig(**h)
    )


def test_scope_level_records_the_threshold_each_level_applied():
    levels = {lvl.name: lvl for lvl in _hierarchy().evaluated_scope.levels}
    assert levels["attr:sex"] == ScopeLevel("attr:sex", ("sex",), {DPD: 0.1})
    # The default multiplier relaxes the intersection to 0.12; the level says so.
    assert levels["intersection:sex_x_race"].thresholds[DPD] == pytest.approx(0.12)
    assert levels["intersection:sex_x_race"].attributes == ("sex", "race")


def test_scope_level_to_dict():
    assert ScopeLevel("attr:sex", ("sex",), {DPD: 0.1}).to_dict() == {
        "name": "attr:sex",
        "attributes": ["sex"],
        "thresholds": {DPD: 0.1},
    }


def test_scope_gap_names_the_setting_that_left_each_group_set_out():
    gaps = _hierarchy(check_single_attributes=False, check_intersections=False).evaluated_scope
    assert [(g.attributes, g.setting) for g in gaps.not_evaluated] == [
        # sex is the overall level's attribute, so it was evaluated; race was not.
        (("race",), "check_single_attributes"),
        (("sex", "race"), "check_intersections"),
    ]
    depth = _hierarchy(intersection_depth=1).evaluated_scope.not_evaluated
    assert [(g.setting, g.reason) for g in depth] == [
        ("intersection_depth", "intersection_depth is 1")
    ]
    assert _hierarchy().evaluated_scope.not_evaluated == ()


def test_scope_gap_to_dict():
    assert ScopeGap(
        ("sex", "race"), "check_intersections", "check_intersections is off"
    ).to_dict() == {
        "attributes": ["sex", "race"],
        "setting": "check_intersections",
        "reason": "check_intersections is off",
    }


def test_evaluated_scope_records_the_configuration_the_levels_ran_with():
    scope = _hierarchy(min_group_size=50).evaluated_scope
    assert isinstance(scope, EvaluatedScope)
    assert scope.attributes_supplied == ("sex", "race")
    assert scope.metrics == (DPD,)
    assert scope.blocking_metrics is None
    # The hierarchy's minimum, which is the one every level certified against.
    assert scope.min_group_size == 50
    flat = _gate().evaluate(np.array([0, 1] * 50), np.array([0, 1] * 50), np.array(["a", "b"] * 50))
    assert flat.evaluated_scope.levels == (ScopeLevel("flat", (), {DPD: 0.1}),)


def test_evaluated_scope_to_dict_is_the_whole_record():
    scope = EvaluatedScope(
        attributes_supplied=("sex", "race"),
        metrics=(DPD,),
        blocking_metrics=(DPD,),
        min_group_size=30,
        levels=(ScopeLevel("attr:sex", ("sex",), {DPD: 0.1}),),
        not_evaluated=(ScopeGap(("sex", "race"), "check_intersections", "off"),),
    )
    assert scope.to_dict() == {
        "attributes_supplied": ["sex", "race"],
        "metrics": [DPD],
        "blocking_metrics": [DPD],
        "min_group_size": 30,
        "levels": [{"name": "attr:sex", "attributes": ["sex"], "thresholds": {DPD: 0.1}}],
        "not_evaluated": [
            {"attributes": ["sex", "race"], "setting": "check_intersections", "reason": "off"}
        ],
    }


# --- ModelFairnessGate.evaluate_against_policy -----------------------------------


def test_evaluate_against_policy_returns_the_unchanged_measurement_and_refuses_a_non_policy():
    sex, race, unfair, _ = _cells()
    hconfig = HierarchicalGateConfig(check_intersections=False)
    held = _decide(unfair, hconfig)
    alone = _gate().evaluate_hierarchical(
        unfair, unfair, {"sex": sex, "race": race}, hierarchical_config=hconfig
    )
    assert isinstance(held, PolicyDecision)
    assert sorted(held.measurement.level_results) == sorted(alone.level_results)
    assert held.measurement.approved is alone.approved is True
    assert held.permitted is False
    with pytest.raises(TypeError, match="must be a CoveragePolicy"):
        _gate().evaluate_against_policy({"policy_id": "p"}, unfair, unfair, {"sex": sex})
