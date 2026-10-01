"""Three states, never two, in every compliance artifact.

Publish-readiness blocker B1. Until 2026-08-28 every reader in
``operations.reporting.compliance`` was ``m.get("passed", True)`` (and, in the
signed test log's snapshot, ``m.get("passed", False)``), so ONE missing
measurement produced TWO opposite fabrications:

* a metric row with no ``passed`` key was counted and printed as **PASS**, with
  a value four times its own threshold in the same row;
* ``.get(key, default)`` does not fire when the key is PRESENT holding ``None``,
  and ``None`` is falsy, so the IDENTICAL unmeasured state printed **FAIL**.

A fabricated all-clear and a fabricated breach, in one table, decided by nothing
but whether a key was absent or ``None``. And ``compute_signed_test_log`` wrote
BOTH readings into ONE document (snapshot default ``False``, summary default
``True``) and hashed the contradiction into a provenance record.

Every test here fails if any of that comes back. The healthy-input tests at the
bottom are the over-correction control: a fully-measured run must grade exactly
as it did before.
"""

import json

import pytest

from vfairness.operations.reporting.compliance import (
    _VERDICT_DERIVED,
    _VERDICT_REPORTED,
    _VERDICT_UNASSESSED,
    _coerce_passed,
    _metric_verdict,
    _metric_verdict_counts,
    compute_signed_test_log,
    generate_annex_iv_data,
    generate_dpia_sections,
    generate_model_card,
    generate_risk_register_from_audit,
)

PROFILE = {
    "name": "Pinned System",
    "purpose": "credit scoring",
    "domain": "credit",
    "version": "1.0",
    "protected_attributes": ["gender"],
}
DATA_REF = {"dataset_name": "loans", "n_records": 1000, "protected_attributes": ["gender"]}

# The audit's exact reproduction: two rows in the SAME unmeasured state, one
# with the key absent and one with the key present holding None.
REPRO_METRICS = {
    "metrics": [
        {"name": "demographic_parity", "value": 0.41, "threshold": 0.10},
        {"name": "equal_opportunity", "value": 0.38, "threshold": 0.10, "passed": None},
    ]
}

# Rows that carry no verdict AND cannot be graded: an unknown better-direction,
# and a row with no numbers at all.
UNGRADABLE_METRICS = {
    "metrics": [
        {"name": "mystery_score", "value": 0.9, "threshold": 0.5},
        {"name": "no_numbers_at_all", "passed": None},
    ]
}

# The over-correction control: every row carries a real boolean verdict.
HEALTHY_METRICS = {
    "metrics": [
        {
            "name": "demographic_parity_difference",
            "value": 0.15,
            "threshold": 0.10,
            "passed": False,
            "severity": 4,
        },
        {
            "name": "equal_opportunity_difference",
            "value": 0.04,
            "threshold": 0.10,
            "passed": True,
            "severity": 2,
        },
        {
            "name": "disparate_impact_ratio",
            "value": 0.92,
            "threshold": 0.80,
            "passed": True,
            "severity": 1,
        },
    ]
}


class TestVerdictReader:
    """The reader itself: absent / None / non-boolean are all could-not-check."""

    @pytest.mark.parametrize(
        "row",
        [
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5},  # key absent
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": None},
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": "yes"},
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": float("nan")},
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": 2},
        ],
    )
    def test_no_usable_verdict_is_never_a_pass_and_never_a_fail(self, row):
        passed, basis, reason = _metric_verdict(row)
        assert passed is None
        assert basis == _VERDICT_UNASSESSED
        assert reason  # it must be able to say WHY

    def test_absent_key_and_none_key_give_the_same_answer(self):
        """The whole defect in one assertion: same state, same verdict."""
        absent = _metric_verdict({"name": "demographic_parity", "value": 0.41, "threshold": 0.10})
        none_held = _metric_verdict(
            {"name": "demographic_parity", "value": 0.41, "threshold": 0.10, "passed": None}
        )
        assert absent == none_held
        assert absent[0] is False  # a 4x breach, and it is not a PASS

    def test_a_reported_boolean_still_wins(self):
        # Even against its own numbers: the pipeline's verdict is not overruled
        # by a comparison performed here.
        assert _metric_verdict({"name": "demographic_parity", "passed": True})[:2] == (
            True,
            _VERDICT_REPORTED,
        )
        assert _metric_verdict(
            {"name": "demographic_parity", "value": 0.41, "threshold": 0.10, "passed": True}
        )[:2] == (True, _VERDICT_REPORTED)

    def test_derivation_respects_the_metric_direction(self):
        """A hand-rolled `value > threshold` would invert the ratio family."""
        # lower-is-better: 0.41 above a 0.10 maximum is a FAIL.
        assert _metric_verdict({"name": "demographic_parity", "value": 0.41, "threshold": 0.10})[
            :2
        ] == (False, _VERDICT_DERIVED)
        # higher-is-better: 0.41 below a 0.80 minimum is ALSO a FAIL, and a
        # naive `value > threshold` would have called it a pass.
        assert _metric_verdict(
            {"name": "disparate_impact_ratio", "value": 0.41, "threshold": 0.80}
        )[:2] == (False, _VERDICT_DERIVED)
        assert _metric_verdict(
            {"name": "disparate_impact_ratio", "value": 0.92, "threshold": 0.80}
        )[:2] == (True, _VERDICT_DERIVED)

    def test_unknown_direction_fails_closed_rather_than_guessing(self):
        assert _metric_verdict({"name": "mystery_score", "value": 0.9, "threshold": 0.5})[0] is None

    def test_a_bound_no_value_can_breach_grades_nothing(self):
        # check_threshold's degenerate-bound guard must reach this module too:
        # a required minimum of 0.0 is met by every possible value.
        assert (
            _metric_verdict({"name": "disparate_impact_ratio", "value": 0.0, "threshold": 0.0})[0]
            is None
        )

    def test_counts_never_fold_the_third_bucket_into_the_other_two(self):
        counts = _metric_verdict_counts(
            REPRO_METRICS["metrics"] + UNGRADABLE_METRICS["metrics"] + HEALTHY_METRICS["metrics"]
        )
        assert counts["passed"] + counts["failed"] + counts["not_assessed"] == counts["total"]
        assert counts["not_assessed"] == 2

    def test_coerce_passed_agrees_with_the_renderer_implementation(self):
        """Pins the duplicate against its source so the two cannot drift."""
        from vfairness.rendering.adapters_validation import _coerce_passed as renderer

        for raw in (
            True,
            False,
            None,
            0,
            1,
            2,
            -1,
            "True",
            "False",
            "",
            float("nan"),
            0.0,
            1.0,
            [],
            object(),
        ):
            assert _coerce_passed(raw) is renderer(raw), raw


class TestModelCard:
    def test_the_reproduction_no_longer_grades_an_unassessed_metric_as_pass(self):
        card = generate_model_card(PROFILE, REPRO_METRICS)
        assert "| demographic_parity | 0.4100 | 0.1000 | PASS |" not in card
        assert "**Metrics passed:** 0" in card
        assert "**Metrics failed:** 2" in card

    def test_ungradable_rows_render_as_their_own_token(self):
        card = generate_model_card(PROFILE, UNGRADABLE_METRICS)
        assert "| mystery_score | 0.9000 | 0.5000 | NOT ASSESSED |" in card
        assert "**Metrics not assessed:** 2" in card
        assert "**Metrics passed:** 0" in card
        assert "**Metrics failed:** 0" in card

    @pytest.mark.parametrize(
        "row",
        [
            # Key ABSENT: the shape that produced the fabricated all-clear,
            # because `not m.get("passed", True)` counted zero failures.
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5},
            # Key PRESENT holding None: the same unmeasured state, which the
            # same expression counted as a failure instead.
            {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": None},
        ],
        ids=["key-absent", "key-none"],
    )
    def test_no_all_clear_above_a_table_of_ungraded_rows(self, row):
        card = generate_model_card(PROFILE, {"metrics": [row]})
        assert "| mystery_score | 0.9000 | 0.5000 | NOT ASSESSED |" in card
        # Neither fabrication is allowed: no all-clear (the absent-key reading)
        # and no breach warning either (the None-holding-key reading). The row
        # was not assessed, and that is the only thing the card may say.
        assert "All evaluated fairness metrics met their thresholds" not in card
        assert "did not meet their threshold" not in card
        assert "**Metrics not assessed:** 1" in card
        assert "**Not assessed:** 1 recorded fairness metric(s)" in card

    def test_a_derived_verdict_is_labelled_as_derived(self):
        card = generate_model_card(PROFILE, REPRO_METRICS)
        assert "FAIL (derived)" in card
        assert "*(derived)*" in card  # the legend explains the token

    def test_healthy_input_is_graded_exactly_as_before(self):
        card = generate_model_card(PROFILE, HEALTHY_METRICS)
        assert "**Metrics passed:** 2" in card
        assert "**Metrics failed:** 1" in card
        assert "**Metrics not assessed:** 0" in card
        assert "| demographic_parity_difference | 0.1500 | 0.1000 | FAIL |" in card
        assert "| equal_opportunity_difference | 0.0400 | 0.1000 | PASS |" in card
        assert "| disparate_impact_ratio | 0.9200 | 0.8000 | PASS |" in card
        # No derived/unassessed vocabulary leaks into a fully-reported card.
        assert "derived" not in card
        assert "NOT ASSESSED" not in card

    def test_healthy_all_pass_keeps_its_all_clear(self):
        all_pass = {"metrics": [dict(m, passed=True) for m in HEALTHY_METRICS["metrics"]]}
        card = generate_model_card(PROFILE, all_pass)
        assert "All evaluated fairness metrics met their thresholds" in card
        assert "**Metrics passed:** 3" in card


class TestDpia:
    def _section_2(self, metric_results):
        dpia = generate_dpia_sections(PROFILE, DATA_REF, [], metric_results)
        return next(s for s in dpia["sections"] if s["section_number"] == 2)["content"]

    def test_art_35_7_b_does_not_claim_an_assessment_that_did_not_happen(self):
        content = self._section_2(UNGRADABLE_METRICS)
        # All three counts pinned in one string. Asserting only "not assessed"
        # left the FAILED count free, and the None-holding row still landed in
        # it: a fabricated breach beside a correct third bucket.
        assert "0 passed, 0 failed, 2 not assessed, out of 2 recorded" in content
        assert "carry no pass/fail verdict" in content

    def test_healthy_counts_are_unchanged(self):
        content = self._section_2(HEALTHY_METRICS)
        assert "2 passed, 1 failed, 0 not assessed, out of 3 recorded" in content
        assert "carry no pass/fail verdict" not in content


class TestSignedTestLog:
    def test_the_document_cannot_contradict_itself(self):
        """The snapshot and the summary must be one reading, not two."""
        log = compute_signed_test_log(
            {"metrics": REPRO_METRICS["metrics"] + UNGRADABLE_METRICS["metrics"]},
            data_hash="d" * 64,
            lib_version="0.1.0",
        )
        snap = log["metrics_snapshot"]
        summary = log["test_results_summary"]
        assert sum(1 for s in snap if s["passed"] is True) == summary["passed"]
        assert sum(1 for s in snap if s["passed"] is False) == summary["failed"]
        assert sum(1 for s in snap if s["passed"] is None) == summary["not_assessed"]

    def test_an_unassessed_row_is_null_in_the_snapshot_not_false(self):
        log = compute_signed_test_log(UNGRADABLE_METRICS, data_hash="d" * 64, lib_version="0.1.0")
        assert [s["passed"] for s in log["metrics_snapshot"]] == [None, None]
        assert all(s["verdict_basis"] == _VERDICT_UNASSESSED for s in log["metrics_snapshot"])
        assert all(s["verdict_note"] for s in log["metrics_snapshot"])

    def test_the_summary_carries_the_unassessed_count(self):
        log = compute_signed_test_log(UNGRADABLE_METRICS, data_hash="d" * 64, lib_version="0.1.0")
        assert log["test_results_summary"]["not_assessed"] == 2
        assert log["test_results_summary"]["passed"] == 0
        assert log["test_results_summary"]["failed"] == 0

    def test_the_hash_seals_the_unassessed_count(self):
        """Move a row out of the third bucket and the content_hash must move."""
        ungraded = compute_signed_test_log(
            {"metrics": [{"name": "mystery_score", "value": 0.9, "threshold": 0.5}]},
            data_hash="d" * 64,
            lib_version="0.1.0",
        )
        graded = compute_signed_test_log(
            {
                "metrics": [
                    {"name": "mystery_score", "value": 0.9, "threshold": 0.5, "passed": True}
                ]
            },
            data_hash="d" * 64,
            lib_version="0.1.0",
        )
        assert ungraded["content_hash"] != graded["content_hash"]
        # And the sealed payload names the state, so a verifier can read it.
        assert ungraded["test_results_summary"]["not_assessed"] == 1
        assert graded["test_results_summary"]["not_assessed"] == 0

    def test_a_missing_overall_verdict_is_not_a_fabricated_failure(self):
        log = compute_signed_test_log({"metrics": []}, data_hash="d" * 64, lib_version="0.1.0")
        assert log["test_results_summary"]["overall_pass"] is None

    def test_the_log_stays_json_serialisable(self):
        log = compute_signed_test_log(UNGRADABLE_METRICS, data_hash="d" * 64, lib_version="0.1.0")
        json.dumps(log, default=str)

    def test_healthy_input_is_summarised_exactly_as_before(self):
        log = compute_signed_test_log(
            {"metrics": HEALTHY_METRICS["metrics"], "overall_pass": False},
            data_hash="d" * 64,
            lib_version="0.1.0",
        )
        assert log["test_results_summary"]["passed"] == 2
        assert log["test_results_summary"]["failed"] == 1
        assert log["test_results_summary"]["not_assessed"] == 0
        assert log["test_results_summary"]["overall_pass"] is False
        assert [s["passed"] for s in log["metrics_snapshot"]] == [False, True, True]
        # A fully-reported row carries no basis marker, so healthy logs keep
        # exactly the shape they had.
        assert all("verdict_basis" not in s for s in log["metrics_snapshot"])


class TestAnnexIv:
    def test_section_6_does_not_report_ungraded_rows_as_tested(self):
        annex = generate_annex_iv_data(PROFILE, UNGRADABLE_METRICS, [])
        s6 = next(s for s in annex["sections"] if s["section_number"] == 6)
        assert s6["content"]["passed"] == 0
        assert s6["content"]["failed"] == 0
        assert s6["content"]["not_assessed"] == 2
        assert s6["content"]["total_evaluated"] == 0
        assert s6["content"]["total_recorded"] == 2
        # "Testing and Validation: auto" over nothing that was graded is the
        # section-level version of the PASS default.
        assert s6["status"] == "partial"

    def test_section_9_reports_the_third_bucket(self):
        annex = generate_annex_iv_data(PROFILE, UNGRADABLE_METRICS, [])
        s9 = next(s for s in annex["sections"] if s["section_number"] == 9)
        assert s9["content"]["fairness_metrics_summary"]["not_assessed"] == 2
        assert s9["content"]["fairness_metrics_summary"]["failed"] == 0

    def test_healthy_input_keeps_section_6_auto_and_its_counts(self):
        annex = generate_annex_iv_data(PROFILE, HEALTHY_METRICS, [])
        s6 = next(s for s in annex["sections"] if s["section_number"] == 6)
        assert s6["status"] == "auto"
        assert s6["content"]["passed"] == 2
        assert s6["content"]["failed"] == 1
        assert s6["content"]["not_assessed"] == 0
        assert s6["content"]["total_evaluated"] == 3


class TestRiskRegister:
    BIAS = {
        "findings": [
            {
                "id": "B001",
                "category": "demographic_parity",
                "description": "Gender disparity",
                "confidence": 0.85,
                "affected_groups": ["female"],
            }
        ]
    }

    def test_an_ungraded_metric_neither_vanishes_nor_raises_severity(self):
        """REWRITTEN 2026-08-28: the old assertion PINNED THE DEFECT.

        It read ``assert entry["severity"] == 3  # the ungraded severity 5 did
        NOT apply``, i.e. it demanded the hardcoded mid-scale default the
        follow-up audit identified as a fabrication. The PROPERTY it was
        reaching for is right and is kept below: an ungraded metric's severity 5
        must not reach this entry. What was wrong is the value it demanded
        INSTEAD. Nothing measured a severity here, so there is none: 3 is not a
        smaller lie than 5, it is the same lie with a different number, and it
        was multiplied into a risk_score and banded.
        """
        metrics = {
            "metrics": [
                # No verdict, and no direction to derive one from.
                {"name": "demographic_parity_mystery", "severity": 5, "passed": None},
            ]
        }
        reg = generate_risk_register_from_audit(self.BIAS, metrics, "credit")
        entry = reg[0]
        assert entry["related_metric_names"] == []  # not counted as a failure
        assert entry["unassessed_metric_names"] == ["demographic_parity_mystery"]
        # The ungraded severity 5 did NOT apply, and neither did a default.
        assert entry["severity"] is None
        assert entry["severity"] != 5
        assert entry["risk_score"] is None
        assert entry["assessment_state"] == "not_assessed"
        assert "could not corroborate" in entry["treatment"]

    def test_healthy_input_scores_exactly_as_before(self):
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                    "severity": 4,
                }
            ]
        }
        reg = generate_risk_register_from_audit(self.BIAS, metrics, "credit")
        assert reg[0]["risk_score"] == 16
        assert reg[0]["severity"] == 4
        assert reg[0]["related_metric_names"] == ["demographic_parity_difference"]
        assert reg[0]["unassessed_metric_names"] == []
        assert reg[0]["treatment"].endswith("post-processing intervention.")
