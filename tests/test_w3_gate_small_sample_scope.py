"""W3: what the small-sample answer COVERS, on the four surfaces that publish it.

``IntersectionalGateDecision.small_sample_check_ran`` means "every supplied
ATTRIBUTE's groups were compared to ``min_group_size``". That is what its docstring
says and what ``evaluate_hierarchical`` computes. The INTERSECTION cells are sized
in a different loop, which never fed it, and the four graded surfaces rendered the
True as a universal claim about the data.

Measured on HEAD 2026-09-30, two attributes whose single-attribute groups are all
100 rows and whose four intersection cells are 95/5/5/95, against a minimum of 30::

    check_intersections=False
        small_sample_check_ran  True
        small_sample_warnings   []
        approved                True
        card                    "- RAN, and every group was at or above the minimum.
                                 This is a measurement, not an absence of one."
        report                  'Small-Sample Check' absent entirely
    check_intersections=True    the SAME data names ('f_y', 5) and ('m_x', 5)
                                against the minimum of 30

So one run asserted a measurement about the very groups the other run found to hold
five people, and the card's own sentence ("This is a measurement, not an absence of
one") is the strongest form that claim can take. The cells of five people are what
intersectional fairness is FOR: a hundred-row attribute can hide them completely.

THE SECOND HALF, same batch: ``to_markdown_report``'s
``elif self.small_sample_check_ran is not True`` rendered NOTHING for the TRUE
state, on BOTH the flat and the hierarchical renderer, while the card printed the
sentence above. Two surfaces answered one question differently and one of the two
answers was silence, which a reader can only decode by knowing that the other two
states print. ``_format_small_sample_state``'s own docstring already states the
rule: "silence is the one answer that is never right."

AFTER: the decision records ``small_sample_unsized``, naming every group set whose
size was never compared and why, at the point the sizing happens or fails to. All
four surfaces carry it, the TRUE sentence is SCOPED whenever anything was left
unsized, and both renderers state the TRUE case in the card's own words.

A SABOTAGE THAT WORKS, for the card rows (recorded 2026-09-29, and it does not
discriminate): re-introducing the pre-fix ``hasattr`` early return

    if not hasattr(self.decision, "small_sample_check_ran"): return []

leaves BOTH pin files GREEN, because IntersectionalGateDecision now HAS that field,
so the branch can never be taken. Measured 2026-09-30:

    hasattr form     -> 35 passed in tests/test_w2_operations_cicd_gate.py
                        20 passed in this file

Only the TYPE-based form reddens, and it is what the code did before the field
existed:

    if isinstance(self.decision, IntersectionalGateDecision): return []
    -> 5 failed, 30 passed in tests/test_w2_operations_cicd_gate.py
       8 failed, 12 passed in this file

Both were executed, and every restore was proven byte-identical with diff -q. The
lesson generalises past this row: a sabotage anchored on the ABSENCE of a thing the
fix ADDED cannot fail, and it looks exactly like a sabotage that passed.

OVER-CORRECTION CONTROLS throughout: the default configuration sizes every group
set it defines, so it must carry NO caveat; a real five-person group must still be
named with its real count; and a three-attribute hierarchy at the default depth of
2 must not start caveating the 3-way cells it never claimed to look at.
"""

from __future__ import annotations

import dataclasses
import json

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
MIN_GROUP = 30
SCOPED = "RAN over every group set it covered"
UNQUALIFIED = "RAN, and every group was at or above the minimum"


def _hidden_cells():
    """Attributes of 100 rows each whose intersection cells are 95/5/5/95.

    The point of the fixture: nothing a single-attribute scan can see is small, and
    two of the four cells hold five people.
    """
    sex = np.array(["f"] * 100 + ["m"] * 100)
    race = np.array(["x"] * 95 + ["y"] * 5 + ["x"] * 5 + ["y"] * 95)
    y = np.array([1, 0] * 100)
    return y, sex, race


def _clean_cells():
    """The same two attributes with 50-row cells, so NOTHING here is small.

    Used wherever the subject is the TRUE state itself: the hidden-cell fixture
    above deliberately contains a five-person cell, which routes the surfaces to
    their warnings branch instead.
    """
    sex = np.array(["f"] * 100 + ["m"] * 100)
    race = np.array((["x"] * 50 + ["y"] * 50) * 2)
    y = np.array([1, 0] * 100)
    return y, sex, race


def _gate(threshold=0.9):
    return ModelFairnessGate(
        config=GateConfig(
            metrics=[METRIC], thresholds={METRIC: threshold}, min_group_size=MIN_GROUP
        )
    )


def _decide(*, check_intersections, attrs=None, **hconfig_kwargs):
    y, sex, race = _hidden_cells()
    protected = attrs if attrs is not None else {"sex": sex, "race": race}
    hconfig = HierarchicalGateConfig(
        check_intersections=check_intersections, min_group_size=MIN_GROUP, **hconfig_kwargs
    )
    return _gate().evaluate_hierarchical(y, y, protected, hierarchical_config=hconfig)


def _surfaces(decision):
    """The four surfaces the audit graded, as a dict of name -> text."""
    card = FairnessReportCard(decision, model_name="loan-v3")
    return {
        "card": card.to_markdown(),
        "card_dict": card.to_dict()["markdown"],
        "github_payload": card.to_github_comment_payload()["body"],
        "report": decision.to_markdown_report(),
    }


# ===========================================================================
# 1. The scope of a True.
# ===========================================================================


class TestTheAnswerCarriesWhatItCovers:
    def test_the_decision_names_the_cells_nothing_sized(self):
        """BEFORE: small_sample_check_ran True, small_sample_warnings [], and no
        field of any kind recording that the four cells were never sized."""
        decision = _decide(check_intersections=False)

        assert decision.small_sample_check_ran is True
        assert decision.small_sample_warnings == []
        assert decision.small_sample_unsized == [
            "the intersection cells of sex_x_race (check_intersections is off)"
        ]

    def test_the_same_data_with_the_level_on_finds_the_five_person_cells(self):
        """The other run, unchanged, and the reason the claim above was false: the
        cells the first run said nothing about hold five people each."""
        decision = _decide(check_intersections=True)

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [
            ("f_y", 5),
            ("m_x", 5),
        ]
        assert all(w.minimum_recommended == MIN_GROUP for w in decision.small_sample_warnings)
        assert decision.small_sample_unsized == []
        assert decision.approved is False

    @pytest.mark.parametrize("surface", ["card", "card_dict", "github_payload", "report"])
    def test_every_surface_states_the_scope(self, surface):
        """BEFORE: the card made the universal claim and the report said nothing at
        all. All four now carry the same two sentences."""
        text = _surfaces(_decide(check_intersections=False))[surface]

        assert SCOPED in text
        assert UNQUALIFIED not in text
        assert "NOT SIZED: the intersection cells of sex_x_race" in text
        assert "check_intersections is off" in text
        assert "nothing above is a statement about those groups" in text

    def test_the_four_surfaces_answer_identically(self):
        """The property the defect is really about: one question, one answer. The
        card and the report differed on the TRUE state by SILENCE, which is not a
        difference a reader can see is a difference."""
        texts = _surfaces(_decide(check_intersections=False))
        sentences = {
            name: [line for line in text.splitlines() if "SIZED" in line or "RAN" in line]
            for name, text in texts.items()
        }

        assert len({tuple(v) for v in sentences.values()}) == 1, sentences
        assert all(len(v) == 2 for v in sentences.values()), sentences

    def test_the_dict_carries_the_scope_through_json(self):
        """A CI integration reads the dict, not the markdown. The 'copy constructor
        drops every later field' shape is how summary_decision was lost once."""
        decision = _decide(check_intersections=False)
        declared = {f.name for f in dataclasses.fields(decision)}
        payload = decision.to_dict()

        assert declared <= set(payload), sorted(declared - set(payload))
        assert payload["small_sample_check_ran"] is True
        assert payload["small_sample_unsized"] == [
            "the intersection cells of sex_x_race (check_intersections is off)"
        ]
        assert "check_intersections is off" in json.dumps(payload, default=str)

    def test_the_cause_is_read_from_the_configuration_that_caused_it(self):
        """Three causes, each naming its own parameter, because the operator's next
        action differs. A depth below 2 defines no intersection at all, and the
        attribute half names the switch that turned the scan off."""
        depth_one = _decide(check_intersections=True, intersection_depth=1)
        scan_off = _decide(check_intersections=True, check_single_attributes=False)

        assert depth_one.small_sample_unsized == [
            "the intersection cells of sex_x_race (intersection_depth is 1)"
        ]
        assert scan_off.small_sample_unsized == [
            "the groups of attribute 'sex' (check_single_attributes is off)",
            "the groups of attribute 'race' (check_single_attributes is off)",
        ]
        assert scan_off.small_sample_check_ran is False

    def test_an_unsized_set_is_disclosed_beside_a_real_warning_too(self):
        """Not only in the clean case. A decision that found a small group AND left
        another set unsized has to say both: the warning is not a substitute for
        the scope, and the scope is not a substitute for the warning."""
        y, sex, race = _hidden_cells()
        sex = np.array(["f"] * 195 + ["m"] * 5)  # a real small single-attribute group

        decision = _gate().evaluate_hierarchical(
            y,
            y,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(
                check_intersections=False, min_group_size=MIN_GROUP
            ),
        )

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("m", 5)]
        for text in _surfaces(decision).values():
            assert "5 samples" in text
            assert "NOT SIZED: the intersection cells of sex_x_race" in text


# ===========================================================================
# 2. The TRUE state is stated, not left as silence, on both renderers.
# ===========================================================================


class TestTheTrueStateIsStatedOnEverySurface:
    def test_the_hierarchical_report_states_it(self):
        """BEFORE: `elif self.small_sample_check_ran is not True` printed nothing,
        so 'Small-Sample Check' was absent from the report of a scan that ran
        clean while the card announced it."""
        y, sex, race = _clean_cells()
        decision = _gate(threshold=0.5).evaluate_hierarchical(y, y, {"sex": sex, "race": race})
        report = decision.to_markdown_report()

        assert decision.small_sample_check_ran is True
        assert decision.small_sample_warnings == []
        assert "## Small-Sample Check" in report
        assert UNQUALIFIED in report

    def test_the_flat_report_states_it_too(self):
        """The SIBLING renderer, carrying the identical `is not True` branch. The
        flat decision from evaluate() is sized over the one attribute it was
        handed, so its claim needs no scope."""
        y = np.array([1, 0] * 100)
        protected = np.array(["a"] * 100 + ["b"] * 100)
        decision = _gate().evaluate(y, y, protected)
        report = decision.to_markdown_report()

        assert decision.small_sample_check_ran is True
        assert "## Small-Sample Check" in report
        assert UNQUALIFIED in report

    def test_the_card_and_the_report_agree_word_for_word(self):
        """Same sentence from the same place, so the two cannot drift apart."""
        y, sex, race = _clean_cells()
        decision = _gate(threshold=0.5).evaluate_hierarchical(y, y, {"sex": sex, "race": race})
        card = FairnessReportCard(decision, "m").to_markdown()

        assert UNQUALIFIED in card
        assert UNQUALIFIED in decision.to_markdown_report()

    def test_the_three_states_are_still_three(self):
        """True / False / not recorded, each with its own sentence, on both the
        card and the report. A hand-built decision has recorded neither."""
        hand_built = IntersectionalGateDecision(approved=True, status=GateStatus.APPROVED)
        # Clean cells, so the group-size scan that DID run found nothing and the
        # surfaces reach their three-state branch rather than their warnings branch.
        y, sex, race = _clean_cells()
        scan_off = _gate(threshold=0.5).evaluate_hierarchical(
            y,
            y,
            {"sex": sex, "race": race},
            hierarchical_config=HierarchicalGateConfig(
                check_single_attributes=False, min_group_size=MIN_GROUP
            ),
        )

        assert "NOT RECORDED" in hand_built.to_markdown_report()
        assert "NOT RECORDED" in FairnessReportCard(hand_built, "m").to_markdown()
        assert hand_built.to_dict()["small_sample_check_ran"] is None

        assert "NOT RUN" in scan_off.to_markdown_report()
        assert "NOT RUN" in FairnessReportCard(scan_off, "m").to_markdown()
        assert scan_off.to_dict()["small_sample_check_ran"] is False

    def test_the_flat_never_checked_decision_is_unchanged(self):
        """evaluate_from_metrics has no sample counts at all, and its NOT RUN
        wording is the one that is true of IT, not of a hierarchy."""
        never = _gate().evaluate_from_metrics({METRIC: 0.2})
        report = never.to_markdown_report()

        assert never.small_sample_check_ran is False
        assert "NOT RUN" in report
        assert "no group sizes" in report
        assert "check_single_attributes" not in report


# ===========================================================================
# 3. OVER-CORRECTION CONTROLS.
# ===========================================================================


class TestControlsTheDisclosureDoesNotSpeakWhenThereIsNothingToSay:
    def test_control_the_default_configuration_carries_no_caveat(self):
        """Both scans on over two attributes: every group set the data defines was
        sized, so the answer needs no scope and the unqualified sentence is true. A
        disclosure that always spoke would be the over-correction."""
        y = np.array([1, 0] * 100)
        sex = np.array(["f"] * 100 + ["m"] * 100)
        race = np.array((["x"] * 50 + ["y"] * 50) * 2)

        decision = _gate(threshold=0.5).evaluate_hierarchical(y, y, {"sex": sex, "race": race})

        assert decision.approved is True
        assert decision.status is GateStatus.APPROVED
        assert decision.small_sample_check_ran is True
        assert decision.small_sample_unsized == []
        for name, text in _surfaces(decision).items():
            assert UNQUALIFIED in text, name
            assert "NOT SIZED" not in text, name
            assert SCOPED not in text, name

    def test_control_the_default_depth_does_not_caveat_a_three_way_cell(self):
        """Three attributes at the default depth of 2. Every PAIR is sized; the
        3-way combination was never in this decision's scope, and listing it would
        turn every default hierarchy into a caveat."""
        y = np.array([1, 0] * 100)
        a = np.array(["p"] * 100 + ["q"] * 100)
        b = np.array((["x"] * 50 + ["y"] * 50) * 2)
        c = np.array((["u"] * 25 + ["v"] * 25) * 4)

        decision = _gate(threshold=0.5).evaluate_hierarchical(y, y, {"a": a, "b": b, "c": c})

        assert sorted(k for k in decision.level_results if k.startswith("intersection")) == [
            "intersection:a_x_b",
            "intersection:a_x_c",
            "intersection:b_x_c",
        ]
        assert decision.small_sample_unsized == []
        assert "NOT SIZED" not in FairnessReportCard(decision, "m").to_markdown()

    def test_control_a_real_small_group_still_reports_its_real_number(self):
        """The measurement is never replaced by the disclosure. Five is still
        five, on every surface, and the comparison is still refused."""
        y = np.array([1, 0] * 50 + [1, 0, 1, 0, 1])
        sex = np.array(["a"] * 100 + ["b"] * 5)

        decision = _gate().evaluate_hierarchical(
            y, y, {"sex": sex}, hierarchical_config=HierarchicalGateConfig(min_group_size=MIN_GROUP)
        )

        assert [(w.group_name, w.sample_size) for w in decision.small_sample_warnings] == [("b", 5)]
        assert decision.small_sample_check_ran is True
        assert decision.small_sample_unsized == []
        assert decision.approved is False
        for name, text in _surfaces(decision).items():
            assert "5 samples" in text, name
            assert "NOT SIZED" not in text, name

    def test_control_one_attribute_defines_no_intersection_to_size(self):
        """With a single attribute an intersection is not DEFINABLE, so nothing was
        left unsized and there is nothing to disclose. Blocking or caveating the
        library's own documented single-attribute call would break the product
        rather than the defect."""
        y = np.array([1, 0] * 100)
        sex = np.array(["f"] * 100 + ["m"] * 100)

        decision = _gate(threshold=0.5).evaluate_hierarchical(
            y,
            y,
            {"sex": sex},
            hierarchical_config=HierarchicalGateConfig(
                check_intersections=False, min_group_size=MIN_GROUP
            ),
        )

        assert decision.small_sample_unsized == []
        assert decision.small_sample_check_ran is True
        assert "NOT SIZED" not in FairnessReportCard(decision, "m").to_markdown()

    def test_control_the_scope_does_not_touch_the_verdict(self):
        """A disclosure is not a refusal. The unsized cells are a caveat on the
        small-sample answer, not a reason to block: the hierarchy that did not look
        at them still approves on the levels it did evaluate, and the levels decide
        the verdict."""
        decision = _decide(check_intersections=False)

        assert decision.small_sample_unsized != []
        assert decision.approved is True
        assert decision.blocking_reasons == []
