"""W2 batch A-operations-1: the six overturned CI/CD gate grades, closed and pinned.

An independent auditor overturned six PROVEN grades in this batch to DEFECT OPEN and
reproduced each one by execution. These are release gates, so they are the highest
severity in the campaign: every one of them was a configured requirement published as
met without being evaluated.

Three root causes, all reproduced on HEAD on 2026-09-29 before any fix.

1. ``ModelFairnessGate.evaluate_from_metrics`` and ``.evaluate`` approved a release
   when a baseline requirement was configured and NO BASELINE VALUE was supplied.
   The guard's precondition read ``baseline_value is None or is_measured(
   baseline_value)``, so an ABSENT baseline was treated as measurable, and
   ``baseline_metrics.get(name) if baseline_metrics else None`` answers None to four
   distinct caller mistakes. Measured with ``GateConfig(metrics=[dp],
   thresholds={dp: 0.9}, require_improvement=True, improvement_margin=0.01)`` and
   ``{dp: 0.2}``, identically through both entry points::

       baseline ABSENT (arg omitted)  approved=True   blocking=0 warns=0  gh 'success'
       baseline dict EMPTY            approved=True   blocking=0 warns=0  gh 'success'
       baseline key MISSING           approved=True   blocking=0 warns=0  gh 'success'
       baseline key present = None    approved=True   blocking=0 warns=0  gh 'success'
       baseline key present = NaN     approved=False  blocking=1          gh 'failure'
       baseline MEASURED (0.3)        approved=True   improvement=0.0999

   all four of the first rows reporting "APPROVED - All fairness requirements met"
   with ``improvement=None``. Fixed with one guard above both comparisons, at BOTH
   call sites, through one shared message builder.

2. ``evaluate_hierarchical``'s coverage guard fired only on ZERO levels, so PARTIAL
   coverage published as a clean pass:

   (a) ``evaluate_hierarchical(y, y, {'gender': 100+100 clean, 'race': 195+5},
       HierarchicalGateConfig(check_single_attributes=False,
       check_intersections=False))`` returned approved=True, "APPROVED - 1/1 levels
       passed", blocking [] warnings [] small_sample_warnings [], and the strings
       'race' and its five-row group were absent from the report AND the card. Level
       1 measures ``attr_names[0]`` only.
   (b) ``check_intersections=True`` with ``intersection_depth=1`` evaluates zero
       intersections, because ``range(2, depth + 1)`` is empty, and returned
       approved=True, "APPROVED - 3/3 levels passed", intersections evaluated [],
       identically for depth 0 and depth -1.

   A SIBLING found while fixing (b): the temporary intersection gate was built from
   a field-by-field kwargs list naming five of GateConfig's seven fields, and
   ``allow_degradation_margin`` is not a constructor parameter at all, so a
   configured degradation bound was NEVER ENFORCED at any intersection level.
   Measured on a gap of 0.5 against a baseline of 0.0 with margin 0.0: overall and
   both single-attribute levels carried margin 0.0, ``intersection:gender_x_race``
   carried None and approved.

3. ``FairnessReportCard.to_markdown`` / ``.to_dict`` / ``.to_github_comment_payload``
   and ``IntersectionalGateDecision.to_markdown_report``: ONE defect on four
   surfaces. ``_format_small_sample_state`` returned ``[]`` for every hierarchical
   decision on a premise written into its own docstring that was false, that the
   check "does not apply" to an IntersectionalGateDecision. It does apply:
   ``evaluate_hierarchical`` runs it, inside ``if hconfig.check_single_attributes:``,
   so the opt-out path laundered the caveat. Measured over two clean groups of 100,
   with the scan off and with it on::

       report 'Small-Sample Check' False   'NOT RUN' False
       card   'Small-Sample Check' False   'NOT RUN' False
       to_dict()['markdown'] any token     False
       to_github_comment_payload()['body'] False
       hasattr(decision, 'small_sample_check_ran')  False

   while the FLAT twin printed "### Small-Sample Check / NOT RUN" on both of its
   surfaces for the same question.

Every cause carries OVER-CORRECTION CONTROLS asserting the healthy case's REAL
number, because a gate that blocks everything passes every refusal test and ships
nothing.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    IntersectionalGateDecision,
    ModelFairnessGate,
)

METRIC = "demographic_parity_difference"
NAN = float("nan")


def _gate(**kwargs):
    return ModelFairnessGate(
        config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.9}, **kwargs)
    )


def _fair_arrays(n: int = 100):
    """Two groups of *n* rows with identical selection rates, so the gap is 0.0."""
    y = np.array([1, 0] * n)
    protected = np.array(["a"] * n + ["b"] * n)
    return y, y, protected


def _two_clean_attrs(n: int = 100):
    """Two attributes, every group 2n/2 rows, so nothing is small and nothing breaches."""
    y = np.array([1, 0] * n)
    gender = np.array(["a"] * n + ["b"] * n)
    race = np.array((["x"] * (n // 2) + ["y"] * (n // 2)) * 2)
    return y, y, gender, race


# ===========================================================================
# CAUSE 1. A baseline requirement with no baseline is not a met requirement.
# ===========================================================================


class TestTheGateRefusesABaselineItWasNeverGiven:
    """The four no-baseline shapes, on both entry points.

    BEFORE (all four, both entry points): approved=True, status=approved,
    "APPROVED - All fairness requirements met", blocking_reasons [], warnings [],
    improvement=None, create_github_check conclusion 'success'.
    """

    @pytest.mark.parametrize(
        "baseline",
        [
            pytest.param("omitted", id="arg_omitted"),
            pytest.param({}, id="dict_empty"),
            pytest.param({"some_other_metric": 0.5}, id="key_missing"),
            pytest.param({METRIC: None}, id="key_present_holding_None"),
        ],
    )
    def test_evaluate_from_metrics_fails_closed_on_every_absent_baseline_shape(self, baseline):
        gate = _gate(require_improvement=True, improvement_margin=0.01)
        kwargs = {} if baseline == "omitted" else {"baseline_metrics": baseline}

        decision = gate.evaluate_from_metrics({METRIC: 0.2}, **kwargs)
        row = decision.metric_evaluations[0]

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        # The measured value is still reported. Only the VERDICT is withheld.
        assert row.value == pytest.approx(0.2)
        assert row.passed is False
        # NaN, not None: the comparison was attempted and could not be made, which
        # is a different state from "no comparison was asked for".
        assert math.isnan(row.improvement), row.improvement
        assert "NO BASELINE VALUE" in row.message
        assert "never checked" in row.message
        assert any("NO BASELINE VALUE" in r for r in decision.blocking_reasons)
        assert gate.create_github_check(decision)["conclusion"] == "failure"

    @pytest.mark.parametrize(
        "baseline",
        [
            pytest.param("omitted", id="arg_omitted"),
            pytest.param({}, id="dict_empty"),
            pytest.param({"some_other_metric": 0.5}, id="key_missing"),
            pytest.param({METRIC: None}, id="key_present_holding_None"),
        ],
    )
    def test_evaluate_fails_closed_on_the_same_four_shapes(self, baseline):
        """THE OTHER CALL SITE. The identical one-line precondition stood in
        evaluate() around gate.py:1063, and this file's whole reason for existing is
        that fixing one of two copies is how this module has been bitten before."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)
        y_true, y_pred, protected = _fair_arrays()
        kwargs = {} if baseline == "omitted" else {"baseline_metrics": baseline}

        decision = gate.evaluate(y_true, y_pred, protected, **kwargs)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert decision.metric_evaluations[0].value == pytest.approx(0.0)
        assert any("NO BASELINE VALUE" in r for r in decision.blocking_reasons)
        assert gate.create_github_check(decision)["conclusion"] == "failure"

    def test_the_degradation_bound_alone_is_enough_to_require_a_baseline(self):
        """THE SIBLING PARAMETER. allow_degradation_margin is configured
        independently of require_improvement and is the OTHER requirement an absent
        baseline switches off. The guard is above both comparisons, so either one on
        its own makes an absent baseline a could-not-check."""
        gate = _gate(allow_degradation_margin=0.0)

        decision = gate.evaluate_from_metrics({METRIC: 0.2})

        assert decision.approved is False
        assert any("NO BASELINE VALUE" in r for r in decision.blocking_reasons)
        # The degradation COMPARISON still did not run; it had no left operand.
        assert not any("degraded" in r for r in decision.blocking_reasons)

    def test_a_non_blocking_metric_downgrades_instead_of_blocking(self):
        """Three states, not two: a non-blocking metric whose baseline was never
        supplied is CONDITIONAL with the reason in warnings and GitHub 'neutral',
        never a green tick and never a block."""
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=[METRIC],
                thresholds={METRIC: 0.9},
                require_improvement=True,
                improvement_margin=0.01,
                blocking_metrics=[],
            )
        )

        decision = gate.evaluate_from_metrics({METRIC: 0.2})

        assert decision.approved is True
        assert decision.status is GateStatus.CONDITIONAL
        assert any("NO BASELINE VALUE" in w for w in decision.warnings)
        assert gate.create_github_check(decision)["conclusion"] == "neutral"

    def test_a_threshold_breach_keeps_its_own_more_precise_reason(self):
        """A metric already failing the comparison must not have its reason replaced
        by the baseline one; the unchecked requirement is still reported, as a
        warning beside the blocking reason."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics({METRIC: 0.95})

        assert decision.approved is False
        assert "exceeds threshold" in decision.blocking_reasons[0]
        assert any("NO BASELINE VALUE" in w for w in decision.warnings)

    def test_the_absent_and_the_non_finite_baseline_are_named_apart(self):
        """Both are could-not-check and both fail closed, but the operator's next
        action differs, so the two sentences must not collapse into one."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        absent = gate.evaluate_from_metrics({METRIC: 0.2}).blocking_reasons[0]
        non_finite = gate.evaluate_from_metrics(
            {METRIC: 0.2}, baseline_metrics={METRIC: NAN}
        ).blocking_reasons[0]

        assert "NO BASELINE VALUE" in absent
        assert "NO BASELINE VALUE" not in non_finite
        assert "could not be measured" in non_finite
        assert absent != non_finite

    # OVER-CORRECTION CONTROLS

    def test_control_a_measured_baseline_still_produces_its_real_improvement(self):
        """THE number, not a shape. A baseline of 0.5 against a measured gap of
        exactly 0.0 is an improvement of exactly 0.5, and the gate approves with a
        green check. A guard that refused every baseline would pass every test
        above and destroy the library."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)
        y_true, y_pred, protected = _fair_arrays()

        decision = gate.evaluate(y_true, y_pred, protected, baseline_metrics={METRIC: 0.5})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.metric_evaluations[0].value == pytest.approx(0.0)
        assert decision.metric_evaluations[0].improvement == pytest.approx(0.5)
        assert gate.create_github_check(decision)["conclusion"] == "success"

    def test_control_no_baseline_requirement_at_all_still_approves(self):
        """THE over-correction control for this cause. With require_improvement=False
        and allow_degradation_margin=None nothing was REQUIRED of the baseline, so an
        absent baseline leaves nothing unchecked: the gate approves, reports the
        measured 0.2, and leaves improvement at None (no comparison was asked for),
        NOT at NaN, so the two states stay distinguishable."""
        gate = _gate()

        decision = gate.evaluate_from_metrics({METRIC: 0.2})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.metric_evaluations[0].value == pytest.approx(0.2)
        assert decision.metric_evaluations[0].improvement is None
        assert decision.blocking_reasons == []
        assert decision.warnings == []
        assert gate.create_github_check(decision)["conclusion"] == "success"

    def test_control_the_requirement_still_bites_when_it_can_be_checked(self):
        """0.2 against a baseline of 0.205 improves by 0.005, under the 0.01 margin,
        and that is still the reason given, not a could-not-check."""
        gate = _gate(require_improvement=True, improvement_margin=0.01)

        decision = gate.evaluate_from_metrics({METRIC: 0.2}, baseline_metrics={METRIC: 0.205})

        assert decision.approved is False
        assert any("did not improve" in r for r in decision.blocking_reasons)
        assert not any("NO BASELINE VALUE" in r for r in decision.blocking_reasons)


# ===========================================================================
# CAUSE 2. Partial coverage is not coverage.
# ===========================================================================


class TestTheHierarchyRefusesWhatItNeverLookedAt:
    def test_an_attribute_supplied_and_never_evaluated_blocks_and_is_named(self):
        """BEFORE: approved=True, "APPROVED - 1/1 levels passed", blocking [],
        warnings [], small_sample_warnings [], and the whole markdown report was
        five content lines in which 'race' appeared nowhere ('race' in card False,
        in report False) although its five-row group was handed to the gate."""
        y = np.array([1, 0] * 100)
        gender = np.array(["a"] * 100 + ["b"] * 100)
        race = np.array(["x"] * 195 + ["y"] * 5)
        hconfig = HierarchicalGateConfig(check_single_attributes=False, check_intersections=False)

        decision = _gate().evaluate_hierarchical(
            y, y, {"gender": gender, "race": race}, hierarchical_config=hconfig
        )

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert list(decision.level_results) == ["overall"]
        reason = " ".join(decision.blocking_reasons)
        assert "never evaluated at any level" in reason
        # The attribute is NAMED. "something was skipped" is not a disclosure.
        assert "'race'" in reason
        # On every surface a reader looks at, not only on the object.
        assert "race" in decision.to_markdown_report()
        assert "race" in FairnessReportCard(decision, "m").to_markdown()
        assert "race" in json.dumps(decision.to_dict(), default=str)

    def test_intersections_asked_for_and_none_evaluated_blocks(self):
        """BEFORE: approved=True, "APPROVED - 3/3 levels passed", intersections
        evaluated [], blocking [] warnings []. `for r in range(2, depth + 1)` is
        EMPTY for any depth below 2, so the whole level silently did nothing."""
        y, _, gender, race = _two_clean_attrs()
        hconfig = HierarchicalGateConfig(check_intersections=True, intersection_depth=1)

        decision = _gate().evaluate_hierarchical(
            y, y, {"gender": gender, "race": race}, hierarchical_config=hconfig
        )

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert [k for k in decision.level_results if k.startswith("intersection:")] == []
        reason = " ".join(decision.blocking_reasons)
        assert "intersection_depth is 1" in reason
        assert "ZERO intersectional groups were evaluated" in reason
        assert "ZERO intersectional groups" in decision.to_markdown_report()
        assert "ZERO intersectional groups" in FairnessReportCard(decision, "m").to_markdown()

    @pytest.mark.parametrize("depth", [0, -1])
    def test_the_boundary_one_step_out_is_refused_too(self, depth):
        """A guard written for depth == 1 would leave depth 0 and negative depths
        open, which is the "== 1 so one value more reopens the defect" shape. All
        three measured identically on HEAD: APPROVED, 3/3 levels passed."""
        y, _, gender, race = _two_clean_attrs()
        hconfig = HierarchicalGateConfig(check_intersections=True, intersection_depth=depth)

        decision = _gate().evaluate_hierarchical(
            y, y, {"gender": gender, "race": race}, hierarchical_config=hconfig
        )

        assert decision.approved is False
        assert f"intersection_depth is {depth}" in " ".join(decision.blocking_reasons)

    def test_one_attribute_with_intersections_on_is_disclosed_not_blocked(self):
        """THE SIBLING CASE, and it is genuinely different, so it is graded
        differently. With one attribute no intersectional group is DEFINABLE, so
        nothing was skipped that could have been done and blocking the library's own
        documented single-attribute call would break the product rather than the
        defect. It is still not silence: the reader who configured intersectional
        checking learns none ran, as a warning, so the decision is CONDITIONAL."""
        y_true, y_pred, protected = _fair_arrays()
        hconfig = HierarchicalGateConfig(check_intersections=True, intersection_depth=2)

        decision = _gate().evaluate_hierarchical(
            y_true, y_pred, {"gender": protected}, hierarchical_config=hconfig
        )

        assert decision.approved is True
        assert decision.status is GateStatus.CONDITIONAL
        assert decision.blocking_reasons == []
        assert any("no intersectional group is definable" in w for w in decision.warnings)
        assert "## Warnings" in decision.to_markdown_report()
        assert "### Warnings" in FairnessReportCard(decision, "m").to_markdown()

    def test_the_intersection_gate_carries_every_configured_requirement(self):
        """THE SIBLING FOUND WHILE FIXING THE ABOVE. The temporary intersection gate
        was rebuilt field by field through the constructor, which has no
        allow_degradation_margin parameter at all, so measured on HEAD:

            overall                    allow_degradation_margin=0.0
            attr:gender                allow_degradation_margin=0.0
            intersection:gender_x_race allow_degradation_margin=None  approved=True

        the intersectional level, the one this method exists for, being the level the
        requirement was switched off on. dataclasses.replace carries every field the
        config has now and every field it gains later."""
        y_true = np.array([1] * 200 + [0] * 200)
        y_pred = np.array([1] * 100 + [0] * 100 + [1] * 100 + [0] * 100)
        gender = np.array(["a"] * 200 + ["b"] * 200)
        race = np.array((["x"] * 100 + ["y"] * 100) * 2)
        gate = _gate(allow_degradation_margin=0.0)
        hconfig = HierarchicalGateConfig(intersection_depth=2, min_group_size=5)

        decision = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"gender": gender, "race": race},
            hierarchical_config=hconfig,
            baseline_metrics={METRIC: 0.0},
        )

        seen = {
            name: level.metadata["config"]["allow_degradation_margin"]
            for name, level in decision.level_results.items()
        }
        assert seen["intersection:gender_x_race"] == 0.0, seen
        # Every field of the parent config reaches every level, not only this one.
        assert set(seen.values()) == {0.0}
        # And the bound actually bites at the intersection: a gap of 0.5 against a
        # baseline of 0.0 is a real degradation past a zero-tolerance margin.
        assert decision.level_results["intersection:gender_x_race"].approved is False

    # OVER-CORRECTION CONTROLS

    def test_control_the_default_config_over_two_attributes_says_nothing(self):
        """The default HierarchicalGateConfig reaches every attribute and evaluates a
        real intersection, so there is nothing to disclose: APPROVED, 4/4 levels, an
        EMPTY blocking_reasons AND an empty warnings list, and no coverage sentence
        on any surface. A guard that always spoke would be the over-correction.

        The base threshold is 0.5 here rather than _gate()'s 0.9 for a reason found
        on 2026-09-30: the default intersectional multiplier of 1.2 relaxes 0.9 to
        1.08, which is OUTSIDE the [0, 1] range of a rate gap, so the intersectional
        level is now refused as a bound no data can breach. That refusal is a
        separate defect with its own pins; this control's subject is the COVERAGE
        disclosure, so it configures a threshold whose relaxed form still grades.
        """
        y, _, gender, race = _two_clean_attrs()

        gate = ModelFairnessGate(config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.5}))
        decision = gate.evaluate_hierarchical(y, y, {"gender": gender, "race": race})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.summary == "APPROVED - 4/4 levels passed"
        assert list(decision.level_results) == [
            "overall",
            "attr:gender",
            "attr:race",
            "intersection:gender_x_race",
        ]
        assert decision.blocking_reasons == []
        assert decision.warnings == []
        assert "never evaluated" not in decision.to_markdown_report()
        assert "ZERO intersectional" not in decision.to_markdown_report()
        assert "never evaluated" not in FairnessReportCard(decision, "m").to_markdown()

    def test_control_an_attribute_reached_only_by_the_intersection_level_is_covered(self):
        """Coverage is read from what RAN, not from the config. With
        check_single_attributes off but intersections on, level 3 reaches BOTH
        attributes, so neither is unevaluated and no coverage reason is produced.

        Threshold 0.5, not _gate()'s 0.9, for the reason recorded on the control
        above: 0.9 x the default 1.2 multiplier is 1.08, a bound outside the range
        of a rate gap, which the intersectional level now refuses on its own terms.
        """
        y, _, gender, race = _two_clean_attrs()
        hconfig = HierarchicalGateConfig(check_single_attributes=False)

        gate = ModelFairnessGate(config=GateConfig(metrics=[METRIC], thresholds={METRIC: 0.5}))
        decision = gate.evaluate_hierarchical(
            y, y, {"gender": gender, "race": race}, hierarchical_config=hconfig
        )

        assert decision.blocking_reasons == []
        assert decision.approved is True
        assert "never evaluated" not in decision.to_markdown_report()

    def test_control_the_zero_level_refusal_is_unchanged(self):
        """The guard this one is the sibling of. Zero levels still produce the
        original sentence, and it is not doubled up with the new ones."""
        y_true, y_pred, _ = _fair_arrays()

        decision = _gate().evaluate_hierarchical(y_true, y_pred, {})

        assert decision.approved is False
        assert any("no level was evaluated" in r for r in decision.blocking_reasons)
        # No attribute was supplied, so there is no unevaluated attribute to name.
        assert not any("never evaluated at any level" in r for r in decision.blocking_reasons)


# ===========================================================================
# CAUSE 3. One defect, four graded surfaces: the small-sample three-state.
# ===========================================================================


def _scan_off():
    y_true, y_pred, protected = _fair_arrays()
    return _gate().evaluate_hierarchical(
        y_true,
        y_pred,
        {"gender": protected},
        hierarchical_config=HierarchicalGateConfig(
            check_single_attributes=False, check_intersections=False
        ),
    )


def _scan_on():
    y_true, y_pred, protected = _fair_arrays()
    return _gate().evaluate_hierarchical(
        y_true,
        y_pred,
        {"gender": protected},
        hierarchical_config=HierarchicalGateConfig(check_intersections=False),
    )


class TestAHierarchyWhoseGroupSizeCheckNeverRanSaysSo:
    def test_the_decision_carries_the_three_state_field(self):
        """BEFORE: hasattr(decision, 'small_sample_check_ran') was False for this
        whole type, which is what the card's early return keyed on."""
        off, on = _scan_off(), _scan_on()

        assert off.small_sample_warnings == []
        assert on.small_sample_warnings == []
        assert off.small_sample_check_ran is False
        assert on.small_sample_check_ran is True
        # Three states. A hand-built decision has recorded neither.
        assert (
            IntersectionalGateDecision(
                approved=True, status=GateStatus.APPROVED
            ).small_sample_check_ran
            is None
        )

    def test_every_declared_field_still_reaches_the_dict(self):
        """The 'copy constructor drops every later field' shape, which is how
        summary_decision was lost once already. The field is worthless if the
        serialiser does not carry it."""
        decision = _scan_off()
        declared = {f.name for f in dataclasses.fields(decision)}

        payload = decision.to_dict()
        assert declared <= set(payload), sorted(declared - set(payload))
        assert payload["small_sample_check_ran"] is False
        assert _scan_on().to_dict()["small_sample_check_ran"] is True
        # Through the JSON round trip a CI integration actually posts.
        assert '"small_sample_check_ran": false' in json.dumps(payload, default=str)

    def test_surface_1_the_hierarchical_markdown_report_names_the_unrun_check(self):
        """BEFORE: 'Small-Sample Check' False, 'NOT RUN' False, while the flat twin
        printed both for the same question. The report is what create_github_check
        publishes as output.text."""
        report = _scan_off().to_markdown_report()

        assert "## Small-Sample Check" in report
        assert "NOT RUN" in report
        assert "did not happen" in report

    def test_surface_2_the_report_card_names_it_too(self):
        """BEFORE: _format_small_sample_state returned [] for every hierarchical
        decision, so this card was byte-identical to the scan-clean one on this
        question. The card is the page a reviewer decides to merge from."""
        markdown = FairnessReportCard(_scan_off(), model_name="loan-v3").to_markdown()

        assert "### Small-Sample Check" in markdown
        assert "NOT RUN" in markdown
        assert "check_single_attributes is off" in markdown

    def test_surface_3_and_4_the_dict_and_the_github_payload_carry_the_same_body(self):
        card = FairnessReportCard(_scan_off(), model_name="loan-v3")

        assert "NOT RUN" in card.to_dict()["markdown"]
        assert "NOT RUN" in card.to_github_comment_payload()["body"]

    def test_the_scan_clean_card_and_the_unrun_card_are_no_longer_identical(self):
        """The property the whole cause is about: the two states have to be
        distinguishable on the surface, not only inside the object."""
        off = FairnessReportCard(_scan_off(), model_name="m").to_markdown()
        on = FairnessReportCard(_scan_on(), model_name="m").to_markdown()

        assert off != on
        assert "NOT RUN" in off
        assert "RAN, and every group was at or above the minimum" in on

    def test_the_hierarchical_not_run_sentence_is_true_of_a_hierarchy(self):
        """The flat card's cause ("made from numbers with no group sizes attached")
        is FALSE of a hierarchy, which was handed the arrays and did not scan them.
        Copying the flat wording across would be a claim stronger than the code
        establishes, which is the failure this section exists to stop."""
        hier = FairnessReportCard(_scan_off()).to_markdown()
        flat = FairnessReportCard(_gate().evaluate_from_metrics({METRIC: 0.2})).to_markdown()

        assert "no group sizes attached" not in hier
        assert "check_single_attributes is off" in hier
        assert "no group sizes attached" in flat
        assert "check_single_attributes" not in flat
        # The invariant half is shared word for word, so the two cannot drift.
        shared = "An empty warning list means the check did not happen, NOT that every"
        assert shared in hier and shared in flat

    def test_a_hand_built_hierarchical_decision_claims_neither(self):
        """Three states, and the removed early return is why this one matters: a
        decision nobody annotated must read NOT RECORDED, not silence."""
        decision = IntersectionalGateDecision(approved=True, status=GateStatus.APPROVED)

        assert "NOT RECORDED" in decision.to_markdown_report()
        assert "NOT RECORDED" in FairnessReportCard(decision, "m").to_markdown()
        assert decision.to_dict()["small_sample_check_ran"] is None

    # OVER-CORRECTION CONTROLS

    def test_control_the_scan_clean_hierarchy_is_not_qualified_on_the_report(self):
        """The scan DID run over every supplied attribute, so its empty list is a
        real finding and the report must not caveat it. THAT is this control's
        subject and it is unchanged.

        The MECHANISM changed in W3 (2026-09-30). This asserted that the report
        printed nothing at all for a True, mirroring the flat twin's
        `elif small_sample_check_ran is not True`. Silence was the defect: the card
        printed "RAN, and every group was at or above the minimum" for the same
        decision, so the two surfaces answered one question differently and the
        report's answer could only be decoded by knowing that the other two states
        print. Both renderers now state the measurement, and "not qualified" is
        asserted as what it means: no caveat token anywhere near it.
        """
        report = _scan_on().to_markdown_report()

        assert "## Small-Sample Check" in report
        assert "RAN, and every group was at or above the minimum" in report
        for caveat in ("NOT RUN", "NOT RECORDED", "NOT SIZED", "cannot be read"):
            assert caveat not in report, caveat
        assert _scan_on().approved is True

    def test_control_a_real_small_group_still_reports_its_size_and_number(self):
        """The check that DID run still names the five-person group, with its real
        count, and still refuses to certify the comparison drawn across it. A
        disclosure that replaced the measurement would be the over-correction."""
        y = np.array([1, 0] * 50 + [1, 0, 1, 0, 1])
        protected = np.array(["a"] * 100 + ["b"] * 5)

        decision = _gate().evaluate_hierarchical(
            y,
            y,
            {"gender": protected},
            hierarchical_config=HierarchicalGateConfig(check_intersections=False),
        )

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 5)]
        assert decision.small_sample_check_ran is True
        assert decision.approved is False
        report = decision.to_markdown_report()
        assert "## Small-Sample Warnings" in report
        assert "5 samples" in report
        # And the card states the same number, not a caveat instead of it.
        card = FairnessReportCard(decision, "m").to_markdown()
        assert "5 samples" in card
        assert "NOT RUN" not in card

    def test_control_the_flat_decisions_three_states_are_unchanged(self):
        """The flat half of all four surfaces was already correct and stays so:
        removing the card's hasattr early return must not have moved it."""
        gate = _gate()
        y_true, y_pred, protected = _fair_arrays()

        checked = gate.evaluate(y_true, y_pred, protected)
        never = gate.evaluate_from_metrics({METRIC: 0.2})

        assert checked.to_dict()["small_sample_check_ran"] is True
        assert never.to_dict()["small_sample_check_ran"] is False
        # W3: the flat report states the TRUE case rather than printing nothing for
        # it, for the reason recorded on the control above. What must not change is
        # that a real measurement is not caveated.
        report = checked.to_markdown_report()
        assert "RAN, and every group was at or above the minimum" in report
        for caveat in ("NOT RUN", "NOT RECORDED", "NOT SIZED", "cannot be read"):
            assert caveat not in report, caveat
        assert "NOT RUN" in never.to_markdown_report()
        assert "NOT RUN" in FairnessReportCard(never).to_markdown()
        assert "RAN, and every group was at or above the minimum" in (
            FairnessReportCard(checked).to_markdown()
        )
