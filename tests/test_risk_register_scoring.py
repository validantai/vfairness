"""An ISO-31000-shaped risk score may not be assembled out of defaults.

Closed 2026-08-28. ``generate_risk_register_from_audit`` invented all three
numbers on the register's headline row for a finding nobody scored:

* ``confidence = finding.get("confidence", 0.5)`` fed
  ``likelihood = round(_clamp(confidence * 5))``, so an unmeasured confidence
  became **likelihood 2** on a 1-5 scale (``round(2.5) == 2``, banker's
  rounding, so the invented midpoint even skewed low);
* ``max_severity = 3`` was a hardcoded default, so every finding this register
  could not corroborate got **severity 3**;
* their product, ``risk_score = 6``, was then banded against the ``>= 15``
  escalation line this module uses in three places, so the fabrication landed
  quietly in ``status: "accepted"`` with "Monitor and reassess at next audit
  cycle." Nothing in the emitted dict said any of the three was invented.

And the two spellings of "unknown" did not even agree: an ABSENT ``confidence``
key fabricated the 6, while a key PRESENT holding ``None`` crashed the report
generator with ``TypeError: unsupported operand type(s) for *: 'NoneType' and
'int'``. One input shape, two different wrong answers.

Three states, never two: assessed / not-assessed, and never a graded number in
between. The last class here is the over-correction control.
"""

import pytest

from vfairness.operations.reporting.compliance import (
    generate_annex_iv_data,
    generate_dpia_sections,
    generate_risk_register_from_audit,
)

PROFILE = {
    "name": "Pinned System",
    "purpose": "credit scoring",
    "domain": "credit",
    "version": "1.0",
    "protected_attributes": ["gender"],
}
DATA_REF = {
    "dataset_name": "loans",
    "n_records": 1000,
    "protected_attributes": ["gender"],
}

#: A real finding that measured no confidence, in both spellings of "unknown".
NO_CONFIDENCE = {
    "findings": [{"id": "B001", "category": "proxy_discrimination", "description": "Proxy risk"}]
}
NONE_CONFIDENCE = {
    "findings": [
        {
            "id": "B001",
            "category": "proxy_discrimination",
            "description": "Proxy risk",
            "confidence": None,
        }
    ]
}

#: The over-correction control: confidence measured, a failed metric with a
#: reported severity. This is the shape the module's own doctest uses.
MEASURED = {
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
MEASURED_METRICS = {
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


class TestAnUnmeasuredConfidenceIsNotALikelihood:
    @pytest.mark.parametrize("bias", [NO_CONFIDENCE, NONE_CONFIDENCE], ids=["absent", "none"])
    def test_no_confidence_yields_no_likelihood_and_no_score(self, bias):
        entry = generate_risk_register_from_audit(bias, {"metrics": []}, "credit")[0]
        assert entry["likelihood"] is None
        assert entry["likelihood"] != 2, "the 0.5 default's likelihood is back"
        assert entry["risk_score"] is None
        assert entry["risk_score"] != 6, "the fabricated 2 x 3 = 6 is back"
        assert entry["residual_risk"] is None

    @pytest.mark.parametrize("bias", [NO_CONFIDENCE, NONE_CONFIDENCE], ids=["absent", "none"])
    def test_both_spellings_of_unknown_take_the_same_path(self, bias):
        """Neither fabricates, and neither raises. The present-but-None variant
        used to die inside ``confidence * 5``."""
        entry = generate_risk_register_from_audit(bias, {"metrics": []}, "credit")[0]
        assert entry["assessment_state"] == "not_assessed"
        assert "likelihood" in entry["not_assessed_fields"]

    @pytest.mark.parametrize("junk", ["0.9", True, float("nan"), object()])
    def test_a_confidence_that_is_not_a_number_is_not_guessed(self, junk):
        bias = {"findings": [{"id": "B001", "category": "proxy", "confidence": junk}]}
        entry = generate_risk_register_from_audit(bias, {"metrics": []}, "credit")[0]
        assert entry["likelihood"] is None
        assert entry["risk_score"] is None

    def test_the_unscored_entry_is_still_on_the_register(self):
        """The FINDING is real; only the grade is withheld. Dropping the row
        would be the opposite fabrication."""
        register = generate_risk_register_from_audit(NO_CONFIDENCE, {"metrics": []}, "credit")
        assert len(register) == 1
        assert register[0]["risk_id"] == "R-001"
        assert register[0]["description"] == "Proxy risk"


class TestAnUnmeasuredSeverityIsNotAThree:
    def test_no_related_failed_metric_means_no_severity(self):
        entry = generate_risk_register_from_audit(MEASURED, {"metrics": []}, "credit")[0]
        assert entry["likelihood"] == 4  # the confidence WAS measured
        assert entry["severity"] is None
        assert entry["severity"] != 3, "the hardcoded mid-scale default is back"
        assert entry["risk_score"] is None

    def test_a_failed_metric_that_reported_no_severity_supplies_none(self):
        """Failing is a measurement. HOW BAD it is, is a separate one, and this
        library cannot derive it from a metric name."""
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                }
            ]
        }
        entry = generate_risk_register_from_audit(MEASURED, metrics, "credit")[0]
        assert entry["related_metric_names"] == ["demographic_parity_difference"]
        assert entry["severity"] is None
        assert entry["risk_score"] is None
        assert "severity" in entry["not_assessed_fields"]

    @pytest.mark.parametrize("reported,expected_score", [(1, 4), (2, 8), (3, 12), (4, 16), (5, 20)])
    def test_a_reported_severity_below_three_is_not_raised_to_three(self, reported, expected_score):
        """A second defect the hardcoded 3 caused, not in the original finding
        and found while fixing it: ``max_severity = 3`` followed by
        ``max(max_severity, ...)`` made 3 a FLOOR, not just a default. A metric
        that reported severity 1 or 2 had it silently RAISED to 3 on the
        register, so the fabrication moved a measured number in the escalating
        direction as well as filling in for an absent one. Reproduced by hand
        against the old two lines: 1 -> 3, 2 -> 3, 3 -> 3, 4 -> 4, 5 -> 5.
        """
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                    "severity": reported,
                }
            ]
        }
        entry = generate_risk_register_from_audit(MEASURED, metrics, "credit")[0]
        assert entry["severity"] == reported
        assert entry["risk_score"] == expected_score

    @pytest.mark.parametrize("junk", ["4", True, float("nan"), None])
    def test_a_severity_that_is_not_a_number_is_not_guessed(self, junk):
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                    "severity": junk,
                }
            ]
        }
        entry = generate_risk_register_from_audit(MEASURED, metrics, "credit")[0]
        assert entry["severity"] is None


class TestAnUnscoredRiskIsNotAccepted:
    def test_status_is_its_own_state_not_accepted(self):
        entry = generate_risk_register_from_audit(NO_CONFIDENCE, {"metrics": []}, "credit")[0]
        assert entry["status"] == "not_assessed"
        assert entry["status"] != "accepted", "an unscored risk was closed as accepted"
        assert entry["treatment"].startswith("NOT SCORED:")
        assert "Monitor and reassess" not in entry["treatment"]

    def test_the_treatment_names_what_was_missing(self):
        entry = generate_risk_register_from_audit(NO_CONFIDENCE, {"metrics": []}, "credit")[0]
        assert "reported no confidence" in entry["treatment"]
        assert "nothing recorded a severity" in entry["treatment"]


class TestConsumersReadTheThirdState:
    """``entry.get("risk_score", 0) >= 15`` is wrong in BOTH directions on an
    unscored entry: the key is present holding None, so the default cannot fire
    and the comparison raises; and the shape it was written for counted as 0,
    i.e. silently below every escalation line."""

    def test_the_dpia_does_not_raise_and_reports_the_unscored_count(self):
        register = generate_risk_register_from_audit(NO_CONFIDENCE, {"metrics": []}, "credit")
        dpia = generate_dpia_sections(PROFILE, DATA_REF, register, {"metrics": []})
        s3 = next(s for s in dpia["sections"] if s["section_number"] == 3)
        assert "Risks that could not be scored: 1" in s3["content"]
        assert "Not scored:" in s3["content"]
        assert "R-001" in s3["content"]
        # The unqualified all-clear may not be written over an unscored register.
        assert "No high-severity risks identified." not in s3["content"]
        assert s3["status"] == "partial"

    def test_annex_iv_does_not_raise_and_reports_the_unscored_count(self):
        register = generate_risk_register_from_audit(NO_CONFIDENCE, {"metrics": []}, "credit")
        annex = generate_annex_iv_data(PROFILE, {"metrics": []}, register)
        s4 = next(s for s in annex["sections"] if s["section_number"] == 4)
        assert s4["content"]["total_risks"] == 1
        assert s4["content"]["high_risks"] == 0
        assert s4["content"]["not_scored"] == 1
        assert s4["status"] == "partial", "an unscorable register is not auto-covered"
        s9 = next(s for s in annex["sections"] if s["section_number"] == 9)
        assert s9["content"]["risk_register_summary"]["not_scored"] == 1


class TestHealthyInputScoresExactlyAsBefore:
    """Over-correction control. A fully measured finding must grade exactly as
    it did before this fix, including the module's own doctest value."""

    def test_the_measured_register_entry_is_unchanged(self):
        entry = generate_risk_register_from_audit(MEASURED, MEASURED_METRICS, "credit")[0]
        assert entry["likelihood"] == 4
        assert entry["severity"] == 4
        assert entry["risk_score"] == 16
        assert entry["residual_risk"] == 12
        assert entry["status"] == "open"
        assert entry["assessment_state"] == "assessed"
        assert entry["not_assessed_fields"] == []
        assert entry["treatment"].endswith("post-processing intervention.")

    def test_a_measured_low_risk_is_still_accepted(self):
        bias = {"findings": [{"id": "B001", "category": "demographic_parity", "confidence": 0.2}]}
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                    "severity": 2,
                }
            ]
        }
        entry = generate_risk_register_from_audit(bias, metrics, "credit")[0]
        assert (entry["likelihood"], entry["severity"], entry["risk_score"]) == (1, 2, 2)
        assert entry["status"] == "accepted"
        assert entry["treatment"] == "Monitor and reassess at next audit cycle."

    def test_a_fully_scored_register_keeps_its_auto_sections(self):
        register = generate_risk_register_from_audit(MEASURED, MEASURED_METRICS, "credit")
        dpia = generate_dpia_sections(PROFILE, DATA_REF, register, MEASURED_METRICS)
        s3 = next(s for s in dpia["sections"] if s["section_number"] == 3)
        assert s3["status"] == "auto"
        assert "Risks that could not be scored: 0" in s3["content"]
        assert "Not scored:" not in s3["content"]
        annex = generate_annex_iv_data(PROFILE, MEASURED_METRICS, register)
        s4 = next(s for s in annex["sections"] if s["section_number"] == 4)
        assert s4["status"] == "auto"
        assert s4["content"]["high_risks"] == 1
        assert s4["content"]["not_scored"] == 0
