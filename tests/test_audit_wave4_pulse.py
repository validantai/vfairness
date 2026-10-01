"""Audit wave 4 pins for the Pulse wording/claims unit.

Each test pins one adversarially confirmed audit fix:

  * llm_probe: the refusal-disparity sentence names the arm with the
    HIGHER refusal rate as the refused one, whichever side of the
    reference it sits on (the signed rateGap can point either way).
  * orchestrator: _proxy_outcome_disparity only claims "confirmed"
    after a Bonferroni correction over the number of worst-vs-best
    comparisons examined; a nominally significant gap that dies under
    the correction is not flagged.
  * recommend: labelFree is the negation of primaryNeedsLabels; it
    never flips True just because ground truth happens to be present.
  * regulatory: the EU AI Act Art. 10 draft fires for EU member-state
    jurisdictions (names and ISO codes), not only for "eu"/"eea".

Every test runs locally, deterministic, no network.
"""

import numpy as np
import pandas as pd

from vfairness.operations.pulse.llm_probe import llm_probe_pulse
from vfairness.operations.pulse.orchestrator import _proxy_outcome_disparity
from vfairness.operations.pulse.recommend import recommend_fairness_definition
from vfairness.operations.pulse.regulatory import _looks_eu, build_art10

# ── llm_probe: refusal-disparity direction ──────────────────────────────────


class _DirectionalRefusalProxy:
    """Refuses whenever the trigger name appears; answers otherwise.

    Sentiment-loaded texts keep the shared BH family permissive enough
    for the refusal comparison to stay family-wise significant.
    """

    def __init__(self, trigger: str) -> None:
        self.trigger = trigger

    def send_prompt(self, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
        if self.trigger in prompt:
            return {
                "text": "I'm sorry, but I cannot help with that "
                "request. This is a terrible, awful, bad "
                "idea and I refuse to assist."
            }
        return {
            "text": "Certainly, happy to help. This is a great, excellent, wonderful candidate."
        }


def _refusal_findings(trigger: str):
    res = llm_probe_pulse(
        {"model_name": "fake"},
        protected_groups={"gender_name": ["Greg", "Jamal"]},
        domain="hiring",
        jurisdiction="us",
        n_runs=5,
        proxy=_DirectionalRefusalProxy(trigger),
    )
    return [f for f in res["data"]["bias"] if f.get("type") == "llm_refusal_rate_disparity"]


def test_refusal_text_names_reference_arm_when_reference_is_refused():
    # Greg is the REFERENCE arm (first value): rateGap is negative, and
    # before the fix the sentence claimed Jamal (0%) was the refused one.
    found = _refusal_findings("Greg")
    assert found, "refusal-disparity finding must fire"
    plain = found[0]["plain"]
    assert found[0]["refusalRates"] == {"Greg": 1.0, "Jamal": 0.0}
    assert "names Greg (100%" in plain
    assert "for Jamal (0%)" in plain


def test_refusal_text_names_comparison_arm_when_that_arm_is_refused():
    # Symmetric case: the non-reference arm is refused; the original
    # wording direction must survive the fix.
    found = _refusal_findings("Jamal")
    assert found, "refusal-disparity finding must fire"
    plain = found[0]["plain"]
    assert found[0]["refusalRates"] == {"Greg": 0.0, "Jamal": 1.0}
    assert "names Jamal (100%" in plain
    assert "for Greg (0%)" in plain


# ── orchestrator: Bonferroni over comparisons examined ──────────────────────


def _rare_level(ones_from, zeros_from):
    """A two-level column whose small level sits below four-fifths of the
    large one under BOTH outcome vectors used below, without its own
    worst-vs-best z coming anywhere near significance."""
    lab = np.array(["common"] * 400, dtype=object)
    lab[np.array(list(ones_from) + list(zeros_from))] = "rare"
    return lab


def _marginal_frame():
    """colA carries a marginal gap (z ~ 2.2, two-sided p ~ 0.027); colB and
    colC each carry a small, UNCONFIRMED gap, so all three enter the
    Bonferroni family.

    READINESS-6, 2026-09-10: colB and colC used to be
    ``np.tile(["x", "y"], n)``, which alternates against the outcome and gives
    a worst/best ratio of exactly 1.00. This fixture's own docstring said they
    were "examined but show no gap", and that was the defect it was pinning
    in: ``examined += 1`` ran BEFORE the four-fifths screen, so two columns on
    which no z-test was ever computed still tripled the Bonferroni divisor.
    Measured with the divisor corrected, colA's marginal gap then came back
    REPORTED at pAdj 0.0269, and this test went red for the right reason.

    The test's real subject is Bonferroni over the family, so the fixture now
    supplies a family: colB and colC have ratios of 0.66 (marginal y) and 0.73
    (strong y), both below the four-fifths screen, so both genuinely enter the
    family and both stay unconfirmed on their own evidence (raw p 0.097 and
    0.245). The divisor is 3 because three z-tests are run.
    """
    n = 200
    col_a = np.array(["g1"] * n + ["g2"] * n)
    y = np.concatenate(
        [
            np.array([1.0] * 100 + [0.0] * 100),  # g1 rate 0.50
            np.array([1.0] * 78 + [0.0] * 122),  # g2 rate 0.39, ratio 0.78
        ]
    )
    work = pd.DataFrame(
        {
            "colA": col_a,
            "colB": _rare_level(range(0, 9), range(300, 321)),
            "colC": _rare_level(range(20, 29), range(330, 351)),
        }
    )
    return work, y


def _confirmed(rows):
    """Rows the scan actually FLAGGED. A sub-0.80 ratio that fails the
    correction is now recorded as `outcome_disparity_unconfirmed` at severity
    info instead of being discarded in silence, so the flagged set has to be
    selected by method rather than by 'anything came back'."""
    return [r for r in rows if r["method"] == "outcome_disparity"]


def test_marginal_gap_not_confirmed_under_bonferroni():
    work, y = _marginal_frame()
    rows = _proxy_outcome_disparity(work, y, ["colA", "colB", "colC"])
    # Uncorrected p ~ 0.027 x 3 tested comparisons ~ 0.08: not confirmed.
    assert _confirmed(rows) == []
    # ...and it is RECORDED as an unconfirmed lead, not dropped.
    unconfirmed = {r["feature"] for r in rows if r["method"] == "outcome_disparity_unconfirmed"}
    assert "colA" in unconfirmed
    assert all(r["comparisonsExamined"] == 3 for r in rows)


def test_marginal_gap_still_flagged_when_single_comparison():
    work, y = _marginal_frame()
    rows = _proxy_outcome_disparity(work, y, ["colA"])
    # With one comparison the correction is a no-op: legacy behaviour.
    assert [r["method"] for r in rows] == ["outcome_disparity"]
    assert rows[0]["comparisonsExamined"] == 1


def test_untested_columns_do_not_inflate_the_bonferroni_divisor():
    """The other half of the same rule. A column that clears the four-fifths
    screen has no hypothesis tested on it, so it must not raise the bar for
    the columns that do."""
    work, y = _marginal_frame()
    work = work.copy()
    # Ratio 1.00 on both levels: screened out, never tested.
    work["benign"] = np.tile(np.array(["x", "y"]), 200)
    rows = _proxy_outcome_disparity(work, y, ["colA", "colB", "colC", "benign"])
    assert all(r["comparisonsExamined"] == 3 for r in rows)
    assert "benign" not in {r["feature"] for r in rows}


def test_strong_gap_survives_correction_and_claim_is_corrected():
    work, _ = _marginal_frame()
    y = np.concatenate(
        [
            np.array([1.0] * 120 + [0.0] * 80),  # 0.60
            np.array([1.0] * 40 + [0.0] * 160),  # 0.20, z ~ 8
        ]
    )
    rows = _proxy_outcome_disparity(work, y, ["colA", "colB", "colC"])
    confirmed = _confirmed(rows)
    assert [r["method"] for r in confirmed] == ["outcome_disparity"]
    row = confirmed[0]
    assert row["feature"] == "colA"
    assert row["comparisonsExamined"] == 3
    assert row["pValueAdjusted"] < 0.05
    assert "Bonferroni-corrected p<0.05 over 3 comparisons" in row["detail"]
    # The old uncorrected claim wording must be gone.
    assert "confirmed, p<0.05" not in row["detail"]


# ── recommend: labelFree vs primaryNeedsLabels ──────────────────────────────


def test_label_dependent_primary_with_ground_truth_is_not_label_free():
    rec = recommend_fairness_definition(domain="lending", jurisdiction="us", has_ground_truth=True)
    assert rec["primaryNeedsLabels"] is True
    assert rec["labelFree"] is False  # was True before the fix
    assert rec["primaryComputableHere"] is True


def test_label_dependent_primary_without_ground_truth_is_not_label_free():
    rec = recommend_fairness_definition(domain="hiring", jurisdiction="us", has_ground_truth=False)
    assert rec["primaryNeedsLabels"] is True
    assert rec["labelFree"] is False
    assert rec["primaryComputableHere"] is False


def test_label_free_primary_stays_label_free():
    rec = recommend_fairness_definition(domain="generic", jurisdiction="", has_ground_truth=False)
    assert rec["primaryNeedsLabels"] is False
    assert rec["labelFree"] is True
    assert rec["primaryComputableHere"] is True


# ── regulatory: EU screen for member states ─────────────────────────────────


def test_looks_eu_accepts_member_state_names_and_codes():
    for j in (
        "Germany",
        "Deutschland",
        "France",
        "Austria",
        "Sweden",
        "DE",
        "de",
        "fr",
        "NL",
        "Republic of Ireland",
    ):
        assert _looks_eu(j), j
    for j in ("us", "US-CA", "united states", "uk", "switzerland", "brazil", "singapore", ""):
        assert not _looks_eu(j), j


def test_art10_draft_fires_for_member_state_jurisdiction():
    blk = build_art10("hiring", "Germany", None, None, 100, 5, {}, {}, {}, [], [])
    assert blk["applicable"] is True
    blk_us = build_art10("hiring", "US", None, None, 100, 5, {}, {}, {}, [], [])
    assert blk_us["applicable"] is False
