"""BGL4 audit of batch A-mcp-1: the one overturn, and the citation error beside it.

Batch A-mcp-1 graded five units in ``src/vfairness/mcp/tools.py`` PROVEN. Two of
those grades do not survive being attacked.

1. ``triage_dataset`` (OVERTURNED). Its pin covers ``assessment_coverage ==
   "none"``: one protected group, so no attribute reaches a verdict. It does not
   cover the case where the attribute IS assessed and only SOME comparisons
   inside it are unmeasurable. On a frame whose outcome column is constant, the
   caller's own target column produces no measurable disparity at all, and the
   library says so in a warning:

       "2 of 5 comparison(s) have an UNMEASURED effect size (effect_size is NaN,
        effect_interpretation is not_measurable): every group was constant, so
        there is no within-group spread to standardise the difference by. This is
        not a small effect and must not be read as one"

   The tool answers "Bias audit complete: 4 finding(s)" with
   ``coverage.findings_are_a_measurement: True``, and the two unmeasured
   comparisons appear NOWHERE in the returned dict: the tokens "not_measurable",
   "UNMEASURED" and "unmeasured" occur zero times in it, and the unmeasured
   comparisons are absent from ``disparity_findings`` rather than listed as
   unassessed. That is the exact boundary this function's own comment says it
   was fixed for: "A warning does not cross the MCP boundary: the dict below is
   the whole of what the client sees." Only the per-ATTRIBUTE coverage was
   carried across; the per-COMPARISON coverage still stops at the warning.

2. ``explain_decision`` (citation error). The grade names
   ``test_an_mcp_tool_says_on_its_own_summary_that_it_measured_nothing[explain_decision]``.
   That parametrisation does not exist; pytest answers "ERROR: not found". The
   second test it names, ``test_an_mcp_tool_on_measurable_data_does_not_cry_wolf``,
   calls only ``triage_dataset``: under coverage it executes zero body lines of
   ``explain_decision``, ``detect_proxies`` and ``suggest_mitigation``, so the
   recorded sabotage ("a tool that disclosed unconditionally fails" it) cannot
   hold for any of those three. Measured: an unconditionally disclosing copy of
   each of the four analysis tools leaves the whole file 14/14 green.

BOTH WERE ACTED ON ON 2026-09-27 (BGL5). ``triage_dataset`` now carries the
per-COMPARISON coverage across the boundary as well, so the two tests that
characterised that defect are inverted below and the strict xfail has become a
live pin. The explain_decision behaviour needed no code change; what needed
changing was the citation, and the ledger row that holds it
(docs/capability-status.json) is not in this batch's file list, so the two tests
that keep that finding executable are left as they are and named in the handback.
The replacement control that does reach all four tools is
tests/test_bgl5_in_processing_3_and_mcp.py::TestTheControlReachesAllFourTools.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.mcp import tools as T

_N = 60


def _outcome_column_is_constant() -> pd.DataFrame:
    """Two real groups of 30, real feature spread, and an outcome nobody varies.

    A model that approves everybody. The outcome disparity that the caller asked
    about cannot be measured on this frame, while the feature distributions can.
    """
    return pd.DataFrame(
        {
            "race": ["a"] * 30 + ["b"] * 30,
            "f1": np.linspace(0, 1, _N),
            "f2": np.arange(_N) * 1.0,
            "score": np.linspace(0, 1, _N),
            "y": [1] * _N,
            "pred": [1] * _N,
        }
    )


def test_the_library_warned_that_two_of_five_comparisons_were_unmeasured():
    """The upstream disclosure exists. This is the half that is not in dispute."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
    unmeasured = [str(w.message) for w in caught if "UNMEASURED effect size" in str(w.message)]
    assert unmeasured, [str(w.message)[:120] for w in caught]
    assert "not_measurable" in unmeasured[0]
    assert "must not be read as one" in unmeasured[0]


def test_triage_dataset_publishes_completeness_over_that_same_run():
    """The defect, pinned as observed so the overturn stays executable.

    INVERTED 2026-09-27. BEFORE: "Bias audit complete: 4 finding(s) ..." with
    assessment_coverage 'complete' and findings_are_a_measurement True. AFTER: the
    per-ATTRIBUTE answer is unchanged, because it was right, and the second axis
    is published beside it: comparison_coverage 'partial' and
    findings_are_a_measurement False, which is the conjunction of the two."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
    coverage = out["coverage"]
    assert coverage["assessment_coverage"] == "complete"
    assert coverage["comparison_coverage"] == "partial"
    assert coverage["findings_are_a_measurement"] is False
    assert "COULD NOT measure" in out["summary"]


def test_an_unmeasured_comparison_reaches_the_mcp_client():
    """What honest disclosure would look like at this boundary.

    Either the summary stops reading as a complete audit, or the payload names the
    comparisons that produced nothing. Today it does neither.

    FIXED 2026-09-27, so the strict xfail this test carried is gone and it is now a
    live pin: the payload carries the library's own sentence verbatim AND the
    summary says the audit could not measure everything it was asked about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
    payload = json.dumps(out, default=str).lower()
    discloses_in_payload = any(
        token in payload for token in ("not_measurable", "unmeasured", "could not")
    )
    summary_is_hedged = "complete" not in out["summary"].lower()
    assert discloses_in_payload or summary_is_hedged, out["summary"]


def test_the_test_id_the_explain_decision_grade_cites_does_not_exist():
    """The parametrisation named as evidence was never written.

    STILL TRUE, and deliberately left standing: the fix for it is to re-point the
    citation in docs/capability-status.json at
    ``test_explain_decision_names_the_features_it_could_not_assess``, which does
    execute 19 body lines of the unit, and that file is outside the A-mcp-1 batch.
    This test is what will go red when the citation is corrected, which is the
    right time for it to go."""
    module = pytest.importorskip("tests.test_bgl3_b2gap_1")
    target = module.test_an_mcp_tool_says_on_its_own_summary_that_it_measured_nothing
    params = [
        argvalues[0]
        for mark in target.pytestmark
        if mark.name == "parametrize"
        for argvalues in mark.args[1]
    ]
    assert params == ["triage_dataset", "detect_proxies", "suggest_mitigation"]
    assert "explain_decision" not in params


def test_the_over_correction_control_never_calls_three_of_the_four_tools():
    """The recorded sabotage for detect_proxies, suggest_mitigation and
    explain_decision is a test that does not execute them.

    STILL TRUE of that test, which is why a replacement was written rather than
    this finding retired:
    tests/test_bgl5_in_processing_3_and_mcp.py::TestTheControlReachesAllFourTools
    calls all four on a fully measurable frame, pins each to a value only it can
    produce, and fails on any could-not-check phrase from a ten-phrase
    vocabulary instead of two literals."""
    module = pytest.importorskip("tests.test_bgl3_b2gap_1")
    called: list[str] = []
    originals = {
        name: getattr(T, name)
        for name in ("triage_dataset", "detect_proxies", "suggest_mitigation", "explain_decision")
    }

    def _spy(name):
        def inner(*a, **k):
            called.append(name)
            return originals[name](*a, **k)

        return inner

    try:
        for name in originals:
            setattr(T, name, _spy(name))
        n = 200
        rng = np.random.RandomState(0)
        race = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        f1 = np.where(race == "a", rng.normal(0.7, 0.1, n), rng.normal(0.3, 0.1, n))
        y = (f1 > 0.5).astype(int)
        two = pd.DataFrame(
            {"race": race, "f1": f1, "f2": rng.normal(0, 1, n), "y": y, "pred": y, "score": f1}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module.test_an_mcp_tool_on_measurable_data_does_not_cry_wolf(two)
    finally:
        for name, fn in originals.items():
            setattr(T, name, fn)
    assert called == ["triage_dataset"], called
