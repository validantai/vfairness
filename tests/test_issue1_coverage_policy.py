"""vfairness issue #1: policy-required coverage before deployment approval.

The gate only knew what it was configured to do, so every way of configuring it
to do less ended in an approval. Measured on main 88e682c (2026-10-24) with the
fixture below, four cells of 100 rows, every single-attribute group at a 0.5
selection rate and the two-way cells at 0.8 / 0.2 / 0.2 / 0.8, one metric with
threshold 0.1::

    none (intersections checked)                       approved False  BLOCKED
    check_intersections=False                          approved True   APPROVED
    second attribute not supplied                      approved True   CONDITIONAL
    threshold 0.7                                      approved True   APPROVED
    blocking_metrics=[]                                approved True   CONDITIONAL
    default_intersection_threshold_multiplier=6.5      approved True   APPROVED
    per_intersection_thresholds 0.7 for the cell       approved True   APPROVED

and on the 95/5/5/95 cells of test_w3_gate_small_sample_scope.py,
``min_group_size=1`` gave APPROVED with ``small_sample_check_ran=False``.

What this file pins:

1. The reviewer's six validation cases, one test each.
2. Every row above as a NEGATIVE fixture: under a policy that requires the
   intersection, none of them is permitted, and each names the setting.
3. CONTROLS: the measurement inside a PolicyDecision is the measurement
   evaluate_hierarchical returns without a policy, a policy-compatible
   configuration IS permitted, and `approved` / CONDITIONAL keep their meaning.
4. A policy that could not be enforced as written is refused when it is built.
5. The four surfaces carry one wording.
"""

import inspect
import json
import re

import numpy as np
import pytest

from vfairness import (
    CoverageItem,
    CoveragePolicy,
    CoverageState,
    EvaluatedScope,
    FairnessReportCard,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    MeasuredOutcome,
    ModelFairnessGate,
    PolicyDecision,
)
from vfairness.exceptions import ConfigurationError
from vfairness.operations.cicd import assess_coverage
from vfairness.operations.cicd.coverage import _fingerprint

DPD = "demographic_parity_difference"
EOD = "equalized_odds_difference"
DIR = "disparate_impact_ratio"


def _cells():
    """Four cells of 100: every single-attribute group at 0.5, the cells at 0.8/0.2/0.2/0.8."""
    sex = np.array(["f"] * 200 + ["m"] * 200)
    race = np.array((["x"] * 100 + ["y"] * 100) * 2)
    rate = {("f", "x"): 0.8, ("f", "y"): 0.2, ("m", "x"): 0.2, ("m", "y"): 0.8}
    unfair = np.array(
        [1 if i % 10 < rate[(s, r)] * 10 else 0 for i, (s, r) in enumerate(zip(sex, race))]
    )
    fair = np.array([i % 2 for i in range(400)])
    return sex, race, unfair, fair


def _small_cells():
    """The 95/5/5/95 cells of test_w3_gate_small_sample_scope.py."""
    sex = np.array(["f"] * 100 + ["m"] * 100)
    race = np.array(["x"] * 95 + ["y"] * 5 + ["x"] * 5 + ["y"] * 95)
    y = np.array([1, 0] * 100)
    return sex, race, y


def _policy(**overrides):
    kwargs = dict(
        policy_id="hiring-eu",
        version="1",
        required_attributes=["sex", "race"],
        required_intersections=[("race", "sex")],
        required_metrics={DPD: 0.1},
        min_group_size=30,
    )
    kwargs.update(overrides)
    return CoveragePolicy(**kwargs)


def _run(y, attrs=None, *, threshold=0.1, blocking=None, min_group=30, policy=None, **h):
    """One run through the policy path. Policy-compatible by default: multiplier 1.0."""
    h.setdefault("default_intersection_threshold_multiplier", 1.0)
    sex, race, _, _ = _cells()
    gate = ModelFairnessGate(
        config=GateConfig(
            metrics=[DPD],
            thresholds={DPD: threshold},
            blocking_metrics=blocking,
            min_group_size=min_group,
        )
    )
    return gate.evaluate_against_policy(
        policy or _policy(),
        y,
        y,
        attrs if attrs is not None else {"sex": sex, "race": race},
        hierarchical_config=HierarchicalGateConfig(min_group_size=min_group, **h),
    )


def _item(decision: PolicyDecision, *attributes) -> CoverageItem:
    (found,) = [i for i in decision.items if set(i.attributes) == set(attributes)]
    return found


def _surfaces(decision: PolicyDecision):
    card = FairnessReportCard(decision, model_name="loan-v3")
    return {
        "report": decision.to_markdown_report(),
        "card": card.to_markdown(),
        "card_dict": card.to_dict()["markdown"],
        "github": ModelFairnessGate().create_github_check(decision)["output"]["text"],
    }


# ===========================================================================
# 1. The reviewer's six validation cases.
# ===========================================================================


class TestTheSixCases:
    def test_case1_complete_coverage_is_permitted(self):
        """All required groups and metrics assessable and passing: permitted."""
        _, _, _, fair = _cells()
        decision = _run(fair)

        assert decision.permitted is True
        assert decision.reasons == []
        assert [(i.label, i.state, i.outcome) for i in decision.items] == [
            ("sex", CoverageState.ASSESSED, MeasuredOutcome.PASS),
            ("race", CoverageState.ASSESSED, MeasuredOutcome.PASS),
            ("sex × race", CoverageState.ASSESSED, MeasuredOutcome.PASS),
        ]
        assert ModelFairnessGate().create_github_check(decision)["conclusion"] == "success"

    def test_case2_a_required_intersection_switched_off_is_not_permitted(self):
        """The issue's own example: every executed level passes, the policy is not met."""
        _, _, unfair, _ = _cells()
        decision = _run(unfair, check_intersections=False)

        # The measurement keeps its meaning, unchanged: nothing blocked on the
        # levels that ran.
        assert decision.measurement.approved is True
        assert decision.measurement.status is GateStatus.APPROVED
        # The policy does not.
        assert decision.permitted is False
        cell = _item(decision, "sex", "race")
        assert cell.state is CoverageState.OMITTED_BY_CONFIGURATION
        assert "check_intersections is off" in cell.reason
        assert cell.outcome is MeasuredOutcome.NOT_MEASURED
        check = ModelFairnessGate().create_github_check(decision)
        assert check["conclusion"] == "failure"
        assert check["output"]["title"] == "Fairness policy hiring-eu v1: not met"

    def test_case3_an_absent_attribute_is_named_and_the_scope_is_not_reduced(self):
        sex, _, _, fair = _cells()
        decision = _run(fair, attrs={"sex": sex})

        # Three required items still, not the one the supplied data allows.
        assert len(decision.items) == 3
        assert _item(decision, "sex").satisfied
        for attributes in (("race",), ("sex", "race")):
            item = _item(decision, *attributes)
            assert item.state is CoverageState.INPUT_ABSENT
            assert "'race' was not supplied" in item.reason
        assert decision.permitted is False

    def test_case4_too_small_a_group_is_insufficient_evidence_not_a_failure(self):
        """Five-person cells that PASS are insufficient evidence; a real breach is a failure."""
        sex, race, y = _small_cells()
        thin = _run(
            y,
            {"sex": sex, "race": race},
            threshold=0.3,
            min_group=1,
            policy=_policy(required_metrics={DPD: 0.3}),
        )
        cell = _item(thin, "sex", "race")
        assert cell.state is CoverageState.INSUFFICIENT_EVIDENCE
        assert cell.outcome is MeasuredOutcome.PASS
        assert cell.undersized == (("f_y", 5), ("m_x", 5))
        # The configured minimum of 1 could flag nothing, and the measurement
        # approves; the policy counted against its own 30.
        assert thin.measurement.approved is True
        assert thin.permitted is False

        _, _, unfair, _ = _cells()
        breach = _item(_run(unfair), "sex", "race")
        assert breach.state is CoverageState.ASSESSED
        assert breach.outcome is MeasuredOutcome.FAIL
        assert breach.reason.startswith("measured breach: 0.6")

    def test_case5_without_a_policy_nothing_changes_and_the_scope_is_explicit(self):
        sex, race, unfair, _ = _cells()
        gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}))
        decision = gate.evaluate_hierarchical(
            unfair,
            unfair,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(check_intersections=False),
        )

        # Existing behaviour, as measured on main before this change.
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        scope = decision.evaluated_scope
        assert isinstance(scope, EvaluatedScope)
        assert [lvl.name for lvl in scope.levels] == ["overall", "attr:sex", "attr:race"]
        assert scope.levels[1].thresholds == {DPD: 0.1}
        assert [(g.attributes, g.setting) for g in scope.not_evaluated] == [
            (("sex", "race"), "check_intersections")
        ]
        as_json = json.loads(json.dumps(decision.to_dict()))
        assert as_json["evaluated_scope"]["not_evaluated"][0]["reason"] == (
            "check_intersections is off"
        )

    def test_case6_no_exception_can_permit_and_the_assessment_is_kept(self):
        """The library takes no exceptions; expiry and revocation live in the integration."""
        for fn in (ModelFairnessGate.evaluate_against_policy, assess_coverage):
            names = set(inspect.signature(fn).parameters)
            for word in ("exception", "waiver", "override", "authority", "approver", "expiry"):
                assert not any(word in n for n in names), (fn.__name__, names)

        _, _, unfair, _ = _cells()
        claimed = {
            "exception": {
                "authority": "Chief Risk Officer",
                "scope": "sex × race",
                "rationale": "accepted",
                "expires": "2099-01-01",
            }
        }
        gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}))
        sex, race, _, _ = _cells()
        plain = gate.evaluate_against_policy(
            _policy(),
            unfair,
            unfair,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(check_intersections=False),
        )
        with_claim = gate.evaluate_against_policy(
            _policy(),
            unfair,
            unfair,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(check_intersections=False),
            model_metadata=claimed,
        )
        assert with_claim.permitted is plain.permitted is False
        assert [i.to_dict() for i in with_claim.items] == [i.to_dict() for i in plain.items]
        # The historical assessment travels with the policy result.
        assert with_claim.to_dict()["measurement"] == with_claim.measurement.to_dict()


# ===========================================================================
# 2. Every measured bypass, as a negative fixture.
# ===========================================================================

_BYPASSES = [
    (
        "check_intersections off",
        {"check_intersections": False},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "check_intersections is off",
    ),
    (
        "intersection_depth 1",
        {"intersection_depth": 1},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "intersection_depth is 1",
    ),
    (
        "threshold 0.7",
        {"threshold": 0.7},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "GateConfig.thresholds sets 0.7",
    ),
    (
        "blocking_metrics empty",
        {"blocking": []},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "GateConfig.blocking_metrics leaves",
    ),
    (
        "multiplier 6.5",
        {"default_intersection_threshold_multiplier": 6.5},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "default_intersection_threshold_multiplier 6.5",
    ),
    (
        "per-intersection 0.7",
        {"per_intersection_thresholds": {"sex_x_race": {DPD: 0.7}}},
        ("sex", "race"),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "per_intersection_thresholds sets 0.7",
    ),
    (
        "race not supplied",
        {"attrs": "sex only"},
        ("race",),
        CoverageState.INPUT_ABSENT,
        "'race' was not supplied",
    ),
    (
        "check_single_attributes off",
        {"check_single_attributes": False},
        ("race",),
        CoverageState.OMITTED_BY_CONFIGURATION,
        "check_single_attributes is off",
    ),
]


@pytest.mark.parametrize(
    "setting, item_attrs, state, named",
    [b[1:] for b in _BYPASSES],
    ids=[b[0] for b in _BYPASSES],
)
def test_every_measured_bypass_is_not_permitted_and_names_its_setting(
    setting, item_attrs, state, named
):
    sex, race, unfair, _ = _cells()
    setting = dict(setting)
    attrs = {"sex": sex} if setting.pop("attrs", None) else None
    decision = _run(unfair, attrs, **setting)

    assert decision.permitted is False
    item = _item(decision, *item_attrs)
    assert item.state is state
    assert named in item.reason
    assert any(named in r for r in decision.reasons)
    assert ModelFairnessGate().create_github_check(decision)["conclusion"] == "failure"


def test_a_configured_minimum_of_one_cannot_hide_five_person_cells():
    sex, race, y = _small_cells()
    decision = _run(
        y,
        {"sex": sex, "race": race},
        threshold=0.3,
        min_group=1,
        policy=_policy(required_metrics={DPD: 0.3}),
    )

    assert decision.measurement.small_sample_check_ran is False
    assert decision.permitted is False
    assert "fewer rows than the policy's minimum of 30 ('f_y' n=5, 'm_x' n=5)" in (
        _item(decision, "sex", "race").reason
    )


def test_the_default_relaxation_counts_as_looser_and_says_how_to_meet_the_policy():
    """The default 1.2 multiplier relaxes 0.1 to 0.12 at the intersection."""
    _, _, _, fair = _cells()
    decision = _run(fair, default_intersection_threshold_multiplier=1.2)

    cell = _item(decision, "sex", "race")
    assert cell.state is CoverageState.OMITTED_BY_CONFIGURATION
    assert cell.outcome is MeasuredOutcome.PASS
    assert "default_intersection_threshold_multiplier 1.2 relaxes" in cell.reason
    assert "to 0.12" in cell.reason


def test_a_required_metric_the_gate_does_not_configure_is_omitted():
    _, _, _, fair = _cells()
    decision = _run(fair, policy=_policy(required_metrics={DPD: 0.1, EOD: 0.1}))

    for item in decision.items:
        if item.metric == EOD:
            assert item.state is CoverageState.OMITTED_BY_CONFIGURATION
            assert "is not one of the gate's configured metrics" in item.reason
    assert decision.permitted is False


def test_a_configured_metric_with_no_threshold_is_omitted():
    sex, race, _, fair = _cells()
    gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={}))
    decision = gate.evaluate_against_policy(
        _policy(),
        fair,
        fair,
        {"sex": sex, "race": race},
        hierarchical_config=HierarchicalGateConfig(default_intersection_threshold_multiplier=1.0),
    )
    assert "no usable threshold" in _item(decision, "sex").reason
    assert decision.permitted is False


def test_a_value_that_could_not_be_measured_is_insufficient_evidence():
    sex, _, _, fair = _cells()
    one_group = np.array(["only"] * 400)
    decision = _run(fair, {"sex": sex, "race": one_group})

    race = _item(decision, "race")
    assert race.state is CoverageState.INSUFFICIENT_EVIDENCE
    assert "could not be measured" in race.reason
    assert race.outcome is MeasuredOutcome.NOT_MEASURED


def test_a_rate_denominator_below_the_policy_minimum_is_insufficient_evidence():
    """A cell of 100 rows with three positive labels: its true-positive rate rests on 3."""
    sex, race, _, _ = _cells()
    y = np.zeros(400, dtype=int)
    cell_fy = np.where((sex == "f") & (race == "y"))[0]
    y[cell_fy[:3]] = 1
    for other in (("f", "x"), ("m", "x"), ("m", "y")):
        idx = np.where((sex == other[0]) & (race == other[1]))[0]
        y[idx[:50]] = 1
    gate = ModelFairnessGate(
        config=GateConfig(metrics=[EOD], thresholds={EOD: 0.1}, min_group_size=2)
    )
    decision = gate.evaluate_against_policy(
        _policy(required_metrics={EOD: 0.1}),
        y,
        y,
        {"sex": sex, "race": race},
        hierarchical_config=HierarchicalGateConfig(
            min_group_size=2, default_intersection_threshold_multiplier=1.0
        ),
    )

    # The configured floor of 2 lets the measurement approve...
    assert decision.measurement.approved is True
    # ...and the policy's floor of 30 does not.
    cell = _item(decision, "sex", "race")
    assert cell.state is CoverageState.INSUFFICIENT_EVIDENCE
    assert "rests on rates measured over fewer rows than the policy's minimum" in cell.reason
    assert any(name.startswith("f_y") and rows == 3 for name, rows in cell.undersized)


def test_numbers_from_a_compute_metrics_fn_carry_a_note_about_what_was_not_counted():
    sex, race, _, fair = _cells()
    gate = ModelFairnessGate(
        config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}),
        compute_metrics_fn=lambda yt, yp, a: {DPD: 0.01},
    )
    decision = gate.evaluate_against_policy(
        _policy(),
        fair,
        fair,
        {"sex": sex, "race": race},
        hierarchical_config=HierarchicalGateConfig(default_intersection_threshold_multiplier=1.0),
    )
    assert decision.permitted is True
    assert all("compute_metrics_fn" in item.note for item in decision.items)
    assert "What the evidence check could not see" in decision.to_markdown_report()
    assert decision.configuration["compute_metrics_fn"].endswith("<lambda>")


def test_a_higher_is_better_metric_is_compared_in_its_own_direction():
    """disparate_impact_ratio: the bound is a MINIMUM, so 0.7 is looser than 0.8."""
    sex, race, _, fair = _cells()

    def run(threshold):
        gate = ModelFairnessGate(config=GateConfig(metrics=[DIR], thresholds={DIR: threshold}))
        return gate.evaluate_against_policy(
            _policy(required_metrics={DIR: 0.8}),
            fair,
            fair,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(
                default_intersection_threshold_multiplier=1.0
            ),
        )

    assert _item(run(0.7), "sex").state is CoverageState.OMITTED_BY_CONFIGURATION
    assert _item(run(0.9), "sex").state is CoverageState.ASSESSED
    assert run(0.9).permitted is True


# ===========================================================================
# 3. Controls: what must NOT change.
# ===========================================================================


def test_control_the_measurement_is_exactly_what_evaluate_hierarchical_returns():
    sex, race, unfair, _ = _cells()
    hconfig = HierarchicalGateConfig(check_intersections=False)
    gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}))
    alone = gate.evaluate_hierarchical(
        unfair, unfair, {"sex": sex, "race": race}, hierarchical_config=hconfig
    )
    held = gate.evaluate_against_policy(
        _policy(), unfair, unfair, {"sex": sex, "race": race}, hierarchical_config=hconfig
    ).measurement

    def stable(d):
        text = json.dumps(d.to_dict(), sort_keys=True, default=str)
        return re.sub(r'"timestamp": "[^"]+"', "", text)

    assert stable(held) == stable(alone)
    assert held.approved is True and held.status is GateStatus.APPROVED


def test_control_conditional_keeps_its_meaning_and_maps_to_a_neutral_check():
    """A non-required metric that could not be checked: permitted, the measurement CONDITIONAL."""
    sex, race, _, fair = _cells()
    gate = ModelFairnessGate(
        config=GateConfig(
            metrics=[DPD, "auroc_parity"],
            thresholds={DPD: 0.1, "auroc_parity": 0.1},
            blocking_metrics=[DPD],
        )
    )
    decision = gate.evaluate_against_policy(
        _policy(),
        fair,
        fair,
        {"sex": sex, "race": race},
        hierarchical_config=HierarchicalGateConfig(default_intersection_threshold_multiplier=1.0),
    )
    assert decision.measurement.status is GateStatus.CONDITIONAL
    assert decision.permitted is True
    assert ModelFairnessGate().create_github_check(decision)["conclusion"] == "neutral"


def test_control_a_blocked_measurement_is_never_permitted():
    """Required items all pass, an extra supplied attribute breaches: not permitted."""
    sex, race, _, fair = _cells()
    region = np.array(["n"] * 200 + ["s"] * 200)
    skewed = fair.copy()
    skewed[:200] = 1
    decision = _run(skewed, {"sex": sex, "race": race, "region": region})
    assert decision.measurement.approved is False
    assert decision.permitted is False
    assert any("the measurement itself is BLOCKED" in r for r in decision.reasons)


def test_control_the_plain_card_is_unchanged():
    sex, race, _, fair = _cells()
    gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}))
    card = FairnessReportCard(
        gate.evaluate_hierarchical(fair, fair, {"sex": sex, "race": race}), model_name="m"
    )
    assert "**Result**: Approved for deployment" in card.to_markdown()
    assert "Deployment policy" not in card.to_markdown()
    assert "policy_permitted" not in card.to_dict()


def test_the_flat_entry_points_record_their_scope():
    _, _, _, fair = _cells()
    gate = ModelFairnessGate(config=GateConfig(metrics=[DPD], thresholds={DPD: 0.1}))
    flat = gate.evaluate(fair, fair, np.array(["a", "b"] * 200))
    assert flat.evaluated_scope.levels[0].name == "flat"
    numbers = gate.evaluate_from_metrics({DPD: 0.01})
    assert numbers.evaluated_scope.levels[0].name == "provided_metrics"
    assert flat.to_dict()["evaluated_scope"]["levels"][0]["thresholds"] == {DPD: 0.1}


# ===========================================================================
# 4. A policy that could not be enforced as written is refused when built.
# ===========================================================================


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"policy_id": " "}, "policy_id"),
        ({"version": ""}, "version"),
        ({"required_attributes": []}, "requires nothing"),
        ({"required_attributes": "race"}, "list or tuple"),
        ({"required_intersections": [("race",)]}, "fewer than two"),
        ({"required_intersections": [("race", "age")]}, "does not list"),
        ({"required_intersections": "race"}, "list of attribute lists"),
        ({"required_metrics": {}}, "at least one metric"),
        ({"required_metrics": {"mystery_score": 0.1}}, "lower or a higher value"),
        ({"required_metrics": {DPD: 1.0}}, "no result can exceed it"),
        ({"required_metrics": {DPD: -0.1}}, "can be met by no value"),
        ({"required_metrics": {DIR: 0.0}}, "met by every value"),
        ({"required_metrics": {DPD: float("nan")}}, "not a finite number"),
        ({"required_metrics": {DPD: True}}, "not a finite number"),
        ({"min_group_size": 1}, "2 or more"),
        ({"min_group_size": True}, "2 or more"),
    ],
)
def test_an_unenforceable_policy_is_refused(overrides, message):
    with pytest.raises(ConfigurationError, match=message):
        _policy(**overrides)


def test_a_policy_must_be_a_coverage_policy():
    sex, race, _, fair = _cells()
    with pytest.raises(TypeError, match="must be a CoveragePolicy"):
        ModelFairnessGate().evaluate_against_policy(
            {"required_attributes": ["sex"]}, fair, fair, {"sex": sex, "race": race}
        )


# ===========================================================================
# 5. Fingerprints, and one wording on every surface.
# ===========================================================================


def test_fingerprints_identify_the_requirement_and_the_settings():
    same = _policy(required_attributes=["race", "sex"], required_intersections=[("sex", "race")])
    assert same.fingerprint == _policy().fingerprint
    assert _policy(version="2").fingerprint != _policy().fingerprint
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", _policy().fingerprint)

    _, _, _, fair = _cells()
    on, off = _run(fair), _run(fair, check_intersections=False)
    assert on.configuration_fingerprint != off.configuration_fingerprint
    assert on.policy_fingerprint == off.policy_fingerprint == _policy().fingerprint
    # A non-finite setting still fingerprints, and differently from a number.
    assert _fingerprint({"t": float("nan")}) != _fingerprint({"t": 0.1})
    with pytest.raises(TypeError, match="cannot fingerprint"):
        _fingerprint({"t": object()})


def test_every_surface_carries_one_wording():
    _, _, unfair, _ = _cells()
    decision = _run(unfair, check_intersections=False)
    surfaces = _surfaces(decision)

    for name, text in surfaces.items():
        assert "**Deployment policy**: `NOT PERMITTED` under `hiring-eu` v1" in text, name
        for reason in decision.reasons:
            assert reason in text, name
        assert "APPROVED there covers only the levels that ran" in text, name
        assert "They are not signatures" in text, name
        assert decision.policy_fingerprint in text, name
        # The measurement approved, and no surface may read that as a deployment.
        assert "Approved for deployment" not in text, name


def test_the_policy_card_keeps_the_measurement_keys_and_adds_the_permission():
    _, _, unfair, _ = _cells()
    card = FairnessReportCard(_run(unfair, check_intersections=False)).to_dict()
    assert card["approved"] is True
    assert card["status"] == "approved"
    assert card["policy_permitted"] is False
    assert (card["policy_id"], card["policy_version"]) == ("hiring-eu", "1")


def test_the_policy_decision_serialises_to_json():
    _, _, _, fair = _cells()
    payload = json.loads(json.dumps(_run(fair).to_dict()))
    assert payload["permitted"] is True
    assert payload["items"][2]["attributes"] == ["sex", "race"]
    assert payload["items"][2]["state"] == "assessed"
    assert payload["configuration"]["hierarchy"]["check_intersections"] is True
