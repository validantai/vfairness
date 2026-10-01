"""W4: the gate's certification floor counted ROWS where the rate needs LABELS.

``ModelFairnessGate`` refuses to certify a comparison drawn across a group holding
fewer than ``min_group_size`` rows. Three of the five metrics it computes do not rest
on the group's rows at all: a true-positive rate lives on the rows whose true label is
1, a false-positive rate on the rows whose label is 0, predictive parity on the rows
the model predicted positive. The floor was compared against
``np.unique(protected_attr)`` counts only, so a group of 100 rows holding ONE positive
label cleared it and its true-positive rate was certified on that one person.

THE SIBLING OF ``_vacuous_min_group_size`` (W3), which closed a minimum nothing could
fall below. This one closes a minimum applied to the wrong count. Same field, same
surfaces, same three states.

Measured on HEAD 2026-09-30, two groups of 100 rows against ``min_group_size=30``,
group a with 50 positives and group b with only a few, every prediction correct so
both true-positive rates are 1.0 and the gap is the textbook perfect-parity 0.0000::

    positives in group b   1  equalized_odds_difference 0.0000 passed=True
        approved=True  status=approved  create_github_check 'success'
        card row '| equalized_odds_difference | 0.0000 | 0.1000 | Pass |'
        small_sample_warnings []   small_sample_check_ran True
        card '- RAN, and every group was at or above the minimum. This is a
              measurement, not an absence of one.'
    positives in group b   2  identical
    positives in group b   5  identical
    positives in group b  29  identical
    group b shrunk to 29 ROWS  approved=False, BLOCKED, warnings [('b', 29)]

so the identical comparison was REFUSED at 29 rows and CERTIFIED on one label, and the
surface that said so in the strongest words available was talking about a count that
had nothing to do with the number printed beside it. The same shape on predictive
parity: group b with ONE predicted positive, which the model got right, gave
predictive_parity_difference 0.0000, approved, zero warnings.

At the intersectional level, same date, four 50-row cells with 25 positives each
except one cell holding ONE::

    BEFORE  approved=True  'APPROVED - 4/4 levels passed'
            intersection:gender_x_race equalized_odds_difference 0.0000 passed=True
            report 'Blocking Issues' False, card names the leg False, payload False
    AFTER   approved=False 'BLOCKED - 3/4 levels passed' with the leg and its count
            named on the report, the card and the GitHub comment payload, while the
            two single-attribute levels (26 and 50 positives) keep their real 0.0000

AND THE SCOPE OF THE ANSWER, for numbers this gate did not compute. With a
``compute_metrics_fn`` the denominators are the caller's business and invisible here,
and all four surfaces still asserted "RAN, and every group was at or above the
minimum. This is a measurement, not an absence of one." Refusing there would be a
fabricated refusal, the mirror of the fabricated certification, so the scope is
disclosed and no verdict moves.

OVER-CORRECTION CONTROLS, all measured in the same runs: a real 0.10 true-positive
rate gap on 50-positive groups is still measured and approved; a
demographic_parity_difference on the SAME one-label data keeps its verdict and its
real 0.5000, because a selection rate is computed over the whole group; a group below
the ROW floor keeps the row floor's own reason and its legs are not named twice; and a
denominator of ZERO keeps the more precise undefined-rate NaN refusal rather than
collecting a second one.

Units: ``ModelFairnessGate.evaluate``, ``ModelFairnessGate.evaluate_hierarchical``,
``FairnessReportCard.to_markdown`` / ``to_dict`` / ``to_github_comment_payload``,
``IntersectionalGateDecision.to_markdown_report``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    GateStatus,
    HierarchicalGateConfig,
    ModelFairnessGate,
)

EOD = "equalized_odds_difference"
PPD = "predictive_parity_difference"
DP = "demographic_parity_difference"
MIN_GROUP = 30

LEG_REFUSAL = "rests on a rate measured over fewer rows than the minimum group size"
TPR_LEG = "the true-positive rate leg, rows whose true label is 1"
UNQUALIFIED = "RAN, and every group was at or above the minimum"
SCOPED = "RAN over every group set it covered"
NOT_SIZED = "NOT SIZED"


def _two_groups(n_positives_in_b: int, reject_b_positives: bool = False):
    """Two groups of 100 rows. Group a holds 50 positives, group b holds few.

    Nothing here is small by the ROW floor: both groups are 100 rows, more than three
    times ``MIN_GROUP``. Every prediction is correct unless ``reject_b_positives``,
    so both true-positive rates are 1.0 and the measured gap is 0.0000.
    """
    a_true = np.array([1] * 50 + [0] * 50)
    a_pred = a_true.copy()
    b_true = np.array([1] * n_positives_in_b + [0] * (100 - n_positives_in_b))
    b_pred = np.zeros(100, dtype=int) if reject_b_positives else b_true.copy()
    return (
        np.concatenate([a_true, b_true]),
        np.concatenate([a_pred, b_pred]),
        np.array(["a"] * 100 + ["b"] * 100),
    )


def _gate(metric=EOD, threshold=0.1, **kwargs):
    return ModelFairnessGate(
        config=GateConfig(
            metrics=[metric], thresholds={metric: threshold}, min_group_size=MIN_GROUP
        ),
        **kwargs,
    )


def _surfaces(decision):
    """Every graded surface that publishes this answer, from one decision."""
    card = FairnessReportCard(decision, "model")
    return {
        "to_markdown": card.to_markdown(),
        "to_dict": card.to_dict()["markdown"],
        "to_github_comment_payload": card.to_github_comment_payload()["body"],
        "decision report": decision.to_markdown_report(),
    }


# ===========================================================================
# The defect
# ===========================================================================


class TestARateCertifiedOnOnePersonIsRefused:
    @pytest.mark.parametrize("n_positives", [1, 2, 5, 29])
    def test_a_true_positive_rate_below_the_floor_is_could_not_check(self, n_positives):
        """The group clears the row floor three times over and the rate does not."""
        y_true, y_pred, attr = _two_groups(n_positives)
        gate = _gate()

        decision = gate.evaluate(y_true, y_pred, attr)
        evaluation = decision.metric_evaluations[0]

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert evaluation.passed is False
        assert LEG_REFUSAL in evaluation.message
        assert f"{TPR_LEG} n={n_positives}" in evaluation.message
        # The measurement is still published, never replaced by the refusal.
        assert evaluation.value == pytest.approx(0.0, abs=1e-12)
        # And the machine-readable field branch protection keys off.
        assert gate.create_github_check(decision)["conclusion"] == "failure"

    def test_the_count_and_the_leg_are_named_on_every_surface(self):
        """One person, named as one person, on the page a reviewer merges from."""
        y_true, y_pred, attr = _two_groups(1)

        decision = _gate().evaluate(y_true, y_pred, attr)

        assert [
            (w.group_name, w.sample_size, w.metric_name) for w in decision.small_sample_warnings
        ] == [(f"b, {TPR_LEG}", 1, EOD)]
        for name, text in _surfaces(decision).items():
            assert TPR_LEG in text, name
            assert "1 samples" in text, name
            assert UNQUALIFIED not in text, name

    def test_predictive_parity_rests_on_the_rows_the_model_selected(self):
        """The sibling denominator: predicted positives, not true labels.

        Group b holds 100 rows and the model predicted exactly ONE of them positive,
        correctly, so its precision is 1.0 and the published gap is 0.0000.
        """
        a_true = np.array([1] * 50 + [0] * 50)
        a_pred = a_true.copy()
        b_true = np.array([1] + [0] * 99)
        b_pred = b_true.copy()
        y_true = np.concatenate([a_true, b_true])
        y_pred = np.concatenate([a_pred, b_pred])
        attr = np.array(["a"] * 100 + ["b"] * 100)

        decision = _gate(metric=PPD).evaluate(y_true, y_pred, attr)
        evaluation = decision.metric_evaluations[0]

        assert decision.approved is False
        assert evaluation.value == pytest.approx(0.0, abs=1e-12)
        assert "rows the model predicted positive n=1" in evaluation.message
        assert "b, rows the model predicted positive" in _surfaces(decision)["to_markdown"]

    def test_a_breach_fabricated_from_one_label_is_disclosed_too(self):
        """The same hole running the other way.

        Group b's single positive is rejected, so its true-positive rate is 0.0 and the
        gap reads 1.0000, the largest a rate gap can be, out of ONE person's label. The
        threshold breach is the more precise reason and keeps the verdict, and the page
        no longer says only that 1.0000 exceeded 0.1000.
        """
        y_true, y_pred, attr = _two_groups(1, reject_b_positives=True)

        decision = _gate().evaluate(y_true, y_pred, attr)

        assert decision.metric_evaluations[0].value == pytest.approx(1.0)
        assert decision.approved is False
        assert decision.blocking_reasons == [
            "equalized_odds_difference (1.0000) exceeds threshold (0.1000)"
        ]
        assert any(LEG_REFUSAL in w for w in decision.warnings)
        for name, text in _surfaces(decision).items():
            assert TPR_LEG in text, name


class TestTheIntersectionalLevelIsRefusedAndSaysWhy:
    def _hierarchy(self):
        """Four 50-row cells, 25 positives each except one cell holding ONE."""
        gender = np.array(["f"] * 100 + ["m"] * 100)
        race = np.array((["x"] * 50 + ["y"] * 50) * 2)
        cell = np.array([f"{g}_{r}" for g, r in zip(gender, race)])
        y_true = np.zeros(200, dtype=int)
        for name in ("f_x", "m_x", "m_y"):
            y_true[np.where(cell == name)[0][:25]] = 1
        y_true[np.where(cell == "f_y")[0][:1]] = 1
        return y_true, y_true.copy(), gender, race

    def test_the_level_this_method_exists_for_is_not_certified_on_one_label(self):
        y_true, y_pred, gender, race = self._hierarchy()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[EOD], thresholds={EOD: 0.1}, min_group_size=20)
        )

        decision = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"gender": gender, "race": race},
            hierarchical_config=HierarchicalGateConfig(min_group_size=20),
        )

        intersection = decision.level_results["intersection:gender_x_race"]
        assert decision.approved is False
        assert decision.summary == "BLOCKED - 3/4 levels passed"
        assert intersection.metric_evaluations[0].passed is False
        assert LEG_REFUSAL in intersection.metric_evaluations[0].message
        # CONTROL, in the same run: the two single-attribute levels hold 26 and 50
        # positives against a minimum of 20, so they keep their real measurement.
        for level in ("attr:gender", "attr:race", "overall"):
            evaluation = decision.level_results[level].metric_evaluations[0]
            assert evaluation.passed is True, level
            assert evaluation.value == pytest.approx(0.0, abs=1e-12), level

    def test_the_reason_reaches_the_report_the_card_and_the_payload(self):
        y_true, y_pred, gender, race = self._hierarchy()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[EOD], thresholds={EOD: 0.1}, min_group_size=20)
        )

        decision = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"gender": gender, "race": race},
            hierarchical_config=HierarchicalGateConfig(min_group_size=20),
        )

        for name, text in _surfaces(decision).items():
            assert LEG_REFUSAL in text, name
            assert TPR_LEG in text, name


class TestTheScopeOfANumberThisGateDidNotCompute:
    def test_a_compute_metrics_fn_leaves_the_denominators_unsized(self):
        """No verdict moves and no surface claims what it did not size."""
        attr = np.array(["a"] * 100 + ["b"] * 100)
        y = np.array([1, 0] * 100)
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.1}, min_group_size=MIN_GROUP),
            compute_metrics_fn=lambda y_true, y_pred, protected: {DP: 0.05},
        )

        decision = gate.evaluate(y, y, attr)

        # The verdict is untouched: a disclosure is not a refusal.
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.metric_evaluations[0].value == pytest.approx(0.05)
        assert decision.small_sample_unsized == [
            "the rows each metric's rate was computed over (a caller-supplied "
            "compute_metrics_fn produced the numbers, so the gate cannot tell which "
            "rows each rate rests on)"
        ]
        assert decision.to_dict()["small_sample_unsized"] == decision.small_sample_unsized
        for name, text in _surfaces(decision).items():
            assert NOT_SIZED in text, name
            assert SCOPED in text, name
            assert UNQUALIFIED not in text, name

    def test_the_hierarchy_answers_the_same_question_the_same_way(self):
        attr = np.array(["a"] * 100 + ["b"] * 100)
        second = np.array((["x"] * 50 + ["y"] * 50) * 2)
        y = np.array([1, 0] * 100)
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.1}, min_group_size=MIN_GROUP),
            compute_metrics_fn=lambda y_true, y_pred, protected: {DP: 0.05},
        )

        decision = gate.evaluate_hierarchical(
            y,
            y,
            {"a1": attr, "a2": second},
            hierarchical_config=HierarchicalGateConfig(min_group_size=MIN_GROUP),
        )

        assert decision.approved is True
        assert decision.small_sample_check_ran is True
        assert any("compute_metrics_fn" in entry for entry in decision.small_sample_unsized)
        for name, text in _surfaces(decision).items():
            assert NOT_SIZED in text, name
            assert UNQUALIFIED not in text, name


# ===========================================================================
# Over-correction controls. A guard that refuses everything passes every
# refusal test above and is useless.
# ===========================================================================


class TestControlsTheHealthyVerdictIsIntact:
    def test_control_a_real_rate_gap_is_measured_and_approved(self):
        """50 positives in each group and a true-positive rate gap of 0.10.

        The number, not merely the absence of an exception: 1.00 against 0.90.
        """
        a_true = np.array([1] * 50 + [0] * 50)
        a_pred = a_true.copy()
        b_true = np.array([1] * 50 + [0] * 50)
        b_pred = np.array([1] * 45 + [0] * 5 + [0] * 50)
        y_true = np.concatenate([a_true, b_true])
        y_pred = np.concatenate([a_pred, b_pred])
        attr = np.array(["a"] * 100 + ["b"] * 100)

        decision = _gate(threshold=0.2).evaluate(y_true, y_pred, attr)

        assert decision.metric_evaluations[0].value == pytest.approx(0.1, abs=1e-9)
        assert decision.metric_evaluations[0].passed is True
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.small_sample_warnings == []
        assert decision.small_sample_unsized == []
        card = FairnessReportCard(decision, "model").to_markdown()
        assert "| equalized_odds_difference | 0.1000 | 0.2000 | Pass |" in card
        assert UNQUALIFIED in card
        assert NOT_SIZED not in card

    def test_control_a_selection_rate_metric_keeps_its_verdict_on_the_same_data(self):
        """PER METRIC, not a blanket refusal.

        The one-label data above, gated on demographic_parity_difference, which is
        computed over every row of the group: 50 of 100 against 1 of 100, a real
        0.4900 gap, and the label counts say nothing about it.
        """
        y_true, y_pred, attr = _two_groups(1)

        decision = _gate(metric=DP, threshold=0.6).evaluate(y_true, y_pred, attr)

        assert decision.metric_evaluations[0].value == pytest.approx(0.49)
        assert decision.metric_evaluations[0].passed is True
        assert decision.approved is True
        assert decision.warnings == []
        assert decision.small_sample_warnings == []

    def test_control_the_row_floor_keeps_its_own_reason_and_says_it_once(self):
        """A group below the ROW floor is already refused, with the more precise
        reason, and its legs are smaller still: two wordings for one fact is how two
        surfaces drift apart."""
        a_true = np.array([1] * 50 + [0] * 50)
        b_true = np.array([1, 1, 0, 0, 0])
        y_true = np.concatenate([a_true, b_true])
        y_pred = y_true.copy()
        attr = np.array(["a"] * 100 + ["b"] * 5)

        decision = _gate().evaluate(y_true, y_pred, attr)

        assert decision.approved is False
        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 5)]
        assert "minimum group size of 30 ('b' n=5)" in decision.blocking_reasons[0]
        assert not any(
            LEG_REFUSAL in text for text in decision.blocking_reasons + decision.warnings
        )

    def test_control_an_undefined_rate_keeps_the_nan_refusal(self):
        """A denominator of ZERO is already could-not-check with a more precise
        reason. One bound must not collect two different refusals."""
        a_true = np.array([1] * 50 + [0] * 50)
        y_true = np.concatenate([a_true, np.zeros(100, dtype=int)])
        y_pred = y_true.copy()
        attr = np.array(["a"] * 100 + ["b"] * 100)

        decision = _gate().evaluate(y_true, y_pred, attr)

        assert np.isnan(decision.metric_evaluations[0].value)
        assert "could not be measured on this data" in decision.blocking_reasons[0]
        assert not any(
            LEG_REFUSAL in text for text in decision.blocking_reasons + decision.warnings
        )

    def test_control_a_vacuous_minimum_still_reports_itself_as_not_run(self):
        """The W3 answer is unchanged: at a minimum of 1 nothing can fall below it,
        the leg floor cannot fire either, and the surfaces say NOT RUN rather than
        claiming a measurement."""
        y_true, y_pred, attr = _two_groups(1)
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[EOD], thresholds={EOD: 0.1}, min_group_size=1)
        )

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.small_sample_check_ran is False
        assert decision.small_sample_vacuous_minimum == 1
        assert decision.small_sample_warnings == []
        for name, text in _surfaces(decision).items():
            assert "NOT RUN: the minimum was 1" in text, name


# ===========================================================================
# The second door, same shape, reached through is_measured rather than through
# the off-by-one: a certification floor that is not a number.
# ===========================================================================


NOT_A_NUMBER = "NOT RUN: the configured minimum was not a number (nan)"


def _one_person_group():
    """One group of 200 and one group of ONE, a selection-rate gap of 0.5000."""
    attr = np.array(["a"] * 200 + ["b"] * 1)
    y_pred = np.array([1] * 180 + [0] * 20 + [0])
    y_true = np.ones(201, dtype=int)
    return y_true, y_pred, attr


class TestAFloorThatIsNotANumberSaysSo:
    def test_a_nan_minimum_is_not_a_measurement_on_any_surface(self):
        """``nan <= 1`` is False, so the W3 vacuity test called a NaN minimum usable
        while ``count < nan`` is False for every group of every dataset."""
        y_true, y_pred, attr = _one_person_group()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.95}, min_group_size=float("nan"))
        )

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.small_sample_warnings == []
        assert decision.small_sample_check_ran is False
        assert np.isnan(decision.small_sample_vacuous_minimum)
        for name, text in _surfaces(decision).items():
            assert NOT_A_NUMBER in text, name
            assert UNQUALIFIED not in text, name

    def test_the_hierarchy_answers_it_the_same_way(self):
        y_true, y_pred, attr = _one_person_group()
        second = np.array(["x"] * 100 + ["y"] * 101)
        gate = ModelFairnessGate(config=GateConfig(metrics=[DP], thresholds={DP: 0.95}))

        decision = gate.evaluate_hierarchical(
            y_true,
            y_pred,
            {"a1": attr, "a2": second},
            hierarchical_config=HierarchicalGateConfig(min_group_size=float("nan")),
        )

        assert decision.small_sample_warnings == []
        assert decision.small_sample_check_ran is False
        assert np.isnan(decision.small_sample_vacuous_minimum)
        for name, text in _surfaces(decision).items():
            assert NOT_A_NUMBER in text, name

    def test_control_an_infinite_minimum_still_blocks_rather_than_reporting_not_run(self):
        """The OPPOSITE bound, and routing it here would turn a refusal into an
        approval: ``count < inf`` is True for every group, so the scan fires for all
        of them and the gate blocks. Measured, not assumed."""
        y_true, y_pred, attr = _one_person_group()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.95}, min_group_size=float("inf"))
        )

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is False
        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [
            ("a", 200),
            ("b", 1),
        ]
        assert decision.small_sample_check_ran is True
        assert decision.small_sample_vacuous_minimum is None

    def test_control_a_usable_minimum_still_names_the_one_person(self):
        """The whole point of the floor, unchanged: at 2 the one-person comparison is
        refused with its real count."""
        y_true, y_pred, attr = _one_person_group()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.95}, min_group_size=2)
        )

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is False
        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 1)]
        assert decision.small_sample_vacuous_minimum is None
        assert "**b** (1 samples, min 2)" in FairnessReportCard(decision, "model").to_markdown()

    def test_control_a_minimum_of_one_keeps_its_own_cause(self):
        """Two causes, two sentences: 'every group holds at least one row' is not
        true of a minimum that is not a number."""
        y_true, y_pred, attr = _one_person_group()
        gate = ModelFairnessGate(
            config=GateConfig(metrics=[DP], thresholds={DP: 0.95}, min_group_size=1)
        )

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.small_sample_vacuous_minimum == 1
        card = FairnessReportCard(decision, "model").to_markdown()
        assert "NOT RUN: the minimum was 1" in card
        assert NOT_A_NUMBER not in card


# ===========================================================================
# The third door: a blocking requirement declared on a metric nobody gates.
# `is_blocking = name in blocking_metrics` is a MEMBERSHIP TEST, so a name that
# is not gated on does not merely do nothing: it makes every gated metric
# non-blocking, and the gate approves a measured breach.
# ===========================================================================


UNMATCHED = "named in blocking_metrics are not in the gate's `metrics` list"


def _breach():
    """A measured demographic parity gap of 0.5000, five times the threshold."""
    attr = np.array(["a"] * 100 + ["b"] * 100)
    y_pred = np.array([1] * 90 + [0] * 10 + [1] * 40 + [0] * 60)
    y_true = np.ones(200, dtype=int)
    return y_true, y_pred, attr


def _typo_gate(blocking):
    return ModelFairnessGate(
        config=GateConfig(metrics=[DP], thresholds={DP: 0.1}, blocking_metrics=blocking)
    )


class TestATypoInBlockingMetricsCannotApproveABreach:
    def test_evaluate_fails_closed_and_names_the_unmatched_name(self):
        y_true, y_pred, attr = _breach()
        gate = _typo_gate(["demographic_parity_diference"])

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert gate.create_github_check(decision)["conclusion"] == "failure"
        reason = next(r for r in decision.blocking_reasons if UNMATCHED in r)
        assert "'demographic_parity_diference'" in reason
        assert "'demographic_parity_difference'" in reason
        # The measurement itself is untouched and still published.
        assert decision.metric_evaluations[0].value == pytest.approx(0.5)
        card = FairnessReportCard(decision, "model").to_markdown()
        assert "**Result**: Deployment blocked" in card
        assert UNMATCHED in card

    def test_evaluate_from_metrics_has_the_same_guard(self):
        """The same one-line precondition at the second entry point, because fixing
        one of two identical copies is how this file has been bitten before."""
        gate = _typo_gate(["demographic_parity_diference"])

        decision = gate.evaluate_from_metrics({DP: 0.5})

        assert decision.approved is False
        assert decision.status is GateStatus.BLOCKED
        assert any(UNMATCHED in r for r in decision.blocking_reasons)

    def test_control_an_empty_list_is_the_warn_only_mode_and_still_approves(self):
        """``blocking_metrics=[]`` names no requirement and is the only way to ask
        for a warn-only gate. Refusing it would delete a documented mode."""
        y_true, y_pred, attr = _breach()
        gate = _typo_gate([])

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is True
        assert decision.status is GateStatus.CONDITIONAL
        assert gate.create_github_check(decision)["conclusion"] == "neutral"
        assert not any(UNMATCHED in w for w in decision.warnings + decision.blocking_reasons)
        assert decision.metric_evaluations[0].value == pytest.approx(0.5)

    @pytest.mark.parametrize("blocking", [None, [DP]])
    def test_control_a_correct_configuration_blocks_on_the_breach_alone(self, blocking):
        y_true, y_pred, attr = _breach()
        gate = _typo_gate(blocking)

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.approved is False
        assert decision.blocking_reasons == [
            "demographic_parity_difference (0.5000) exceeds threshold (0.1000)"
        ]

    @pytest.mark.parametrize("blocking", [None, [DP], []])
    def test_control_a_passing_model_is_still_approved(self, blocking):
        """The healthy verdict with its real number: a 0.0500 gap against 0.1000."""
        attr = np.array(["a"] * 100 + ["b"] * 100)
        y_pred = np.array([1] * 50 + [0] * 50 + [1] * 45 + [0] * 55)
        y_true = np.ones(200, dtype=int)
        gate = _typo_gate(blocking)

        decision = gate.evaluate(y_true, y_pred, attr)

        assert decision.metric_evaluations[0].value == pytest.approx(0.05)
        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.blocking_reasons == []
        assert gate.create_github_check(decision)["conclusion"] == "success"
