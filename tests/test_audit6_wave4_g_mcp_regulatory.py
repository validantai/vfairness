"""Audit 6, wave 4, lane G: the MCP triage tool and the Pulse regulatory pack.

Two findings of the same shape are pinned here, plus one sibling:

* S-13 ``vfairness.mcp.tools.triage_dataset`` accepted ``target_column`` and
  never passed it on, so ``BiasDetector`` fell back to keyword auto-discovery.
  The caller's real target was then tested as an incidental feature
  DISTRIBUTION rather than as an OUTCOME, and the intersectional outcome
  analysis (which only ever runs on the outcome columns) did not run at all.
* S-14 ``build_art10`` read ``jurisdiction`` and never read ``domain``, so the
  EU AI Act Art. 10 draft came out byte-identical for a hiring model, a credit
  model and a nonsense domain.
* Sibling in ``build_ll144``: with no protected column present at all, the
  unknown-demographics count was reported as 0, a clean-looking measurement of
  "no unknown rows" where no demographic column had been read.

Every pin comes with an over-correction control asserting the MEASURED values
still arrive: a fix that forced an outcome column on every caller, or that
made the Art. 10 draft domain-specific by inventing text, or that turned every
LL144 count into null, would fail those controls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.mcp import tools
from vfairness.operations.pulse.regulatory import build_art10, build_ll144

# ── S-13 fixtures ───────────────────────────────────────────────────────────

# Deterministic frame (no RNG). "hired" is the caller's target and contains no
# keyword the detector's auto-discovery recognises ("outcome", "score", "label",
# "target", ...), so the two paths genuinely diverge. Selection rates by cell:
# m/north 0.85, m/south 0.55, f/north 0.45, f/south 0.15.
_CELL_HIRED = {("m", "north"): 51, ("m", "south"): 33, ("f", "north"): 27, ("f", "south"): 9}


@pytest.fixture()
def triage_frame() -> pd.DataFrame:
    rows = []
    for (gender, region), n_hired in _CELL_HIRED.items():
        for i in range(60):
            rows.append(
                {
                    "gender": gender,
                    "region": region,
                    "hired": 1 if i < n_hired else 0,
                    "tenure": float(i % 12),
                }
            )
    return pd.DataFrame(rows)


def _disparity_rows(result):
    return sorted(
        (f["feature"], f["disparity_type"], f["protected_attribute"])
        for f in result["report"]["disparity_findings"]
    )


# ── S-13: refusal pin ───────────────────────────────────────────────────────


def test_triage_dataset_passes_target_column_through_as_outcome(triage_frame):
    """REFUSAL PIN. The named target must reach BiasDetector as outcome_column.

    Pinned by the findings themselves, not by the echo: the target must be
    tested as an OUTCOME disparity for every protected attribute, and the
    intersectional outcome analysis must run on it. Dropping the argument
    turns both outcome rows into "distribution" rows and deletes the
    intersectional row entirely.
    """
    out = tools.triage_dataset(triage_frame, ["gender", "region"], target_column="hired")

    assert _disparity_rows(out) == [
        ("hired", "intersectional", "Intersection(gender, region)"),
        ("hired", "outcome", "gender"),
        ("hired", "outcome", "region"),
    ]
    assert out["finding_counts"]["disparity_findings"] == 3
    # No finding may downgrade the caller's own target to an incidental
    # feature-distribution difference.
    assert not [
        f
        for f in out["report"]["disparity_findings"]
        if f["feature"] == "hired" and f["disparity_type"] == "distribution"
    ]
    # Provenance, on the tool result and inside the report the caller reads.
    assert out["outcome_column"] == "hired"
    assert out["report"]["dataset_info"]["outcome_column"] == "hired"
    assert "'hired'" in out["summary"]


def test_triage_dataset_rejects_a_target_column_that_is_not_in_the_frame(triage_frame):
    """REFUSAL PIN. A named-but-absent target must raise, not be dropped.

    Without the pass-through this argument was ignored, so a typo produced a
    confident report built on an auto-discovered column the caller never asked
    for. BiasDetector itself would also skip it silently.
    """
    with pytest.raises(ValueError, match="hird"):
        tools.triage_dataset(triage_frame, ["gender", "region"], target_column="hird")


# ── S-13: over-correction controls ──────────────────────────────────────────


def test_triage_dataset_without_a_target_still_returns_the_auto_discovery_answer(triage_frame):
    """OVER-CORRECTION CONTROL. Callers who name no target keep the old path.

    The fix must not invent an outcome column. With target_column omitted the
    detector's own auto-discovery answer stands (2 distribution rows, no
    intersectional outcome row), the echo is None rather than a guess, and the
    summary says which path ran.
    """
    out = tools.triage_dataset(triage_frame, ["gender", "region"])

    assert _disparity_rows(out) == [
        ("hired", "distribution", "gender"),
        ("hired", "distribution", "region"),
    ]
    assert out["finding_counts"]["disparity_findings"] == 2
    assert out["outcome_column"] is None
    assert out["report"]["dataset_info"]["outcome_column"] is None
    assert "auto-discovery" in out["summary"]


def test_triage_dataset_measured_effect_sizes_are_unchanged(triage_frame):
    """OVER-CORRECTION CONTROL. The numbers are the real measured numbers.

    Exact values for this deterministic frame, not membership of a broad set:
    gender 0.391667, region 0.291667, the gender x region intersection 0.5,
    each with its own p-value below 1e-5. The other audit modules must keep
    reporting their real counts too.
    """
    out = tools.triage_dataset(triage_frame, ["gender", "region"], target_column="hired")
    by_attr = {f["protected_attribute"]: f for f in out["report"]["disparity_findings"]}

    assert by_attr["gender"]["effect_size"] == pytest.approx(0.391667, abs=1e-5)
    assert by_attr["gender"]["pvalue"] == pytest.approx(1.29776e-09, rel=1e-3)
    assert by_attr["region"]["effect_size"] == pytest.approx(0.291667, abs=1e-5)
    assert by_attr["region"]["pvalue"] == pytest.approx(6.2285e-06, rel=1e-3)
    assert by_attr["Intersection(gender, region)"]["effect_size"] == pytest.approx(0.5, abs=1e-5)

    assert out["finding_counts"]["representation_findings"] == 2
    assert out["finding_counts"]["proxy_findings"] == 2
    assert out["finding_counts"]["historical_findings"] == 0
    assert out["summary"].startswith("Bias audit complete: 7 finding(s)")


# ── S-14 fixtures ───────────────────────────────────────────────────────────

_ART10_RUN = dict(
    jurisdiction="Germany",
    artifact_label="loans-2026Q1.csv",
    artifact_hash="abc123def",
    rows=1000,
    columns=12,
    schema={"pii_leakage": ["email"], "oracle_columns": [], "model_output": ["pred"]},
    data_quality={"headline": "3% missing in income"},
    verdict={"headline": "Disparity found.", "summary": "Selection rates differ by gender."},
    bias_findings=[{"plain": "Women selected at 0.42x the male rate."}],
    interventions=[{"title": "Reweighing", "how": "resample training data"}],
)


def _exam(block) -> str:
    return [s for s in block["sections"] if s["id"] == "bias_examination"][0]["body"]


# ── S-14: refusal pin ───────────────────────────────────────────────────────


def test_art10_examination_varies_with_the_declared_domain():
    """REFUSAL PIN. The Art. 10 draft must not read the same for every domain.

    Before the fix these three blocks were byte-identical. The examination
    section now carries the G-31 historical-discrimination catalog entry for
    the declared domain, with that entry's own citations, and each domain's
    citations must be absent from the others.
    """
    hiring = build_art10(domain="hiring", **_ART10_RUN)
    credit = build_art10(domain="credit scoring", **_ART10_RUN)
    health = build_art10(domain="healthcare triage", **_ART10_RUN)

    assert _exam(hiring) != _exam(credit) != _exam(health)
    assert _exam(hiring) != _exam(health)

    assert hiring["domainContext"]["state"] == "catalog_match"
    assert hiring["domainContext"]["catalogDomain"] == "hiring"
    assert credit["domainContext"]["catalogDomain"] == "lending"
    assert health["domainContext"]["catalogDomain"] == "healthcare"

    assert "Bertrand and Mullainathan" in _exam(hiring)
    assert "Griggs v. Duke Power" in _exam(hiring)
    assert "Bertrand and Mullainathan" not in _exam(credit)

    assert "Bartlett" in _exam(credit)
    assert "redlining" in _exam(credit).lower()
    assert "Bartlett" not in _exam(hiring)

    assert "Obermeyer" in _exam(health)
    assert "Obermeyer" not in _exam(hiring)

    # The citations are the catalog's, echoed on the block for a reader who
    # wants the sources without re-parsing prose.
    assert any("Bertrand" in c for c in hiring["domainContext"]["citations"])
    assert any("Bartlett" in c for c in credit["domainContext"]["citations"])


def test_art10_uncatalogued_domain_is_disclosed_not_invented():
    """REFUSAL PIN. An unknown domain gets a plain statement, never prose.

    Three states, never two: this is the second one. No catalog entry exists,
    so the draft says it is domain independent and carries no domain citations,
    rather than borrowing another domain's history or inventing new text.
    """
    blk = build_art10(domain="quantum widget pricing", **_ART10_RUN)

    assert blk["applicable"] is True
    assert blk["domainContext"] == {
        "state": "declared_no_catalog_entry",
        "declaredDomain": "quantum widget pricing",
        "catalogDomain": None,
        "citations": [],
    }
    body = _exam(blk)
    assert "no domain-specific examination material" in body
    assert "domain independent" in body
    assert "quantum widget pricing" in body
    for token in ("Bertrand", "Bartlett", "Obermeyer", "Griggs", "redlining"):
        assert token not in body


def test_art10_absent_domain_is_a_third_state():
    """REFUSAL PIN. "No domain declared" must not collapse into "unknown domain".

    The third state. A run that declared nothing is a different fact from a run
    that declared something the catalog does not cover, and the block says
    which one happened.
    """
    blk = build_art10(domain="", **_ART10_RUN)

    assert blk["domainContext"]["state"] == "not_declared"
    assert blk["domainContext"]["declaredDomain"] is None
    assert blk["domainContext"]["citations"] == []
    assert "No domain was declared for this run" in _exam(blk)
    assert build_art10(domain=None, **_ART10_RUN)["domainContext"]["state"] == "not_declared"


# ── S-14: over-correction controls ──────────────────────────────────────────


def test_art10_keeps_every_measured_value_from_the_run():
    """OVER-CORRECTION CONTROL. Domain wiring must not displace the run's data.

    Exact measured values, in the sections that carry them: the artifact
    identity and shape, the schema screen's own column names, the verdict text
    and top finding, and the intervention. All four section ids and their order
    are unchanged.
    """
    blk = build_art10(domain="hiring", **_ART10_RUN)
    by_id = {s["id"]: s["body"] for s in blk["sections"]}

    assert [s["id"] for s in blk["sections"]] == [
        "data_sources",
        "data_governance",
        "bias_examination",
        "bias_mitigation",
    ]
    assert "loans-2026Q1.csv" in by_id["data_sources"]
    assert "content hash abc123def" in by_id["data_sources"]
    assert "1000 rows and 12 columns" in by_id["data_sources"]
    assert "email" in by_id["data_governance"]
    assert "pred" in by_id["data_governance"]
    assert "3% missing in income" in by_id["data_governance"]
    assert by_id["bias_examination"].startswith(
        "Disparity found. Selection rates differ by gender. "
        "Top findings: Women selected at 0.42x the male rate."
    )
    assert "Reweighing: resample training data" in by_id["bias_mitigation"]


def test_art10_jurisdiction_gate_is_untouched():
    """OVER-CORRECTION CONTROL. The EU gate still refuses a non-EU run.

    A hiring domain now has catalog material, which must not talk the block
    into producing an EU draft outside the EU.
    """
    blk = build_art10(domain="hiring", **{**_ART10_RUN, "jurisdiction": "US"})
    assert blk["applicable"] is False
    assert blk["sections"] == []
    assert "does not read as EU" in blk["reason"]
    assert "domainContext" not in blk


# ── Sibling: LL144 unknown-demographics count ───────────────────────────────


def test_ll144_unknown_demographics_is_null_when_nothing_was_measured():
    """REFUSAL PIN (sibling). No protected column means no count, not zero.

    0 reads as "we looked and found no rows with unknown demographics". With
    no protected column in the frame nothing was looked at, so the field is
    null and a note says why, matching empty_exports' own convention.
    """
    df = pd.DataFrame({"age": [30, 40, 50], "y": [1, 0, 1]})
    out = build_ll144(df, [], np.array([1.0, 0.0, 1.0]), df, [], "hiring", "New York", score=None)

    assert out["unknownDemographics"] is None
    assert any("not measured" in n for n in out["notes"])


def test_ll144_unknown_demographics_counts_real_missing_rows():
    """OVER-CORRECTION CONTROL. A real count is still a real count.

    Exact measured values: 2 rows carry a missing gender, and the two known
    groups keep their measured selection rates and impact ratio.
    """
    df = pd.DataFrame(
        {
            "gender": ["m", "m", "m", "m", "f", "f", "f", "f", None, None],
            "y": [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        }
    )
    y_pred = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    out = build_ll144(df, ["gender"], y_pred, df, ["gender"], "hiring", "New York City", score=None)

    assert out["applicable"] is True
    assert out["unknownDemographics"] == 2
    assert not any("not measured" in n for n in out["notes"])
    rates = {
        r["group"]: (r["n"], r["selectionRate"], r["impactRatio"]) for r in out["impactRatios"]
    }
    assert rates["m"] == (4, 0.75, 1.0)
    assert rates["f"][0] == 4
    assert rates["f"][1] == pytest.approx(0.25)
    assert rates["f"][2] == pytest.approx(1.0 / 3.0)
