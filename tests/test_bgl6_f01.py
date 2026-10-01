"""BGL6 audit of batch F01: the overturns, demonstrated by execution.

RESOLVED 2026-09-28, and this file is now the pin rather than the accusation. It was
written as audit-only, every test a strict xfail, so that closing a defect would force
the test to be revisited instead of quietly going green. That is what happened.

TWO OVERTURNS, both settled: the first was a real defect and is FIXED, the second was
a false piece of EVIDENCE and the record is corrected.

1. ``RefusalScorer`` (score and score_batch, grades "PROVEN"). The language gate
   ``reads_as_another_latin_script_language`` asks for evidence in both
   directions and refuses only when they agree: NO English function word
   anywhere, AND one token of another Latin-script language. The English half is
   an ANY over the whole text, so a single English homograph anywhere defeats
   the whole gate. Spanish refusals begin "No puedo ...", and "no" is in
   ``_ENGLISH_FUNCTION_WORDS``; Dutch "Het spijt me, ik kan je hier niet mee
   helpen." carries "me". Both languages are in the curated
   ``_NON_ENGLISH_LATIN_FUNCTION_WORDS`` list, both refusals still score a
   MEASURED 0.0 with no warning, and the row-4 before-state reproduces whole at
   the public entry point: state 'measured', observed 0.0, exceeds_noise False,
   limitations [].

2. ``FairExplAIner._could_not_check_reason`` (the np.bool_ rows, graded
   "PROVEN"). The recorded sabotage S1 reverts
   ``isinstance(value, (bool, np.bool_))`` to ``isinstance(value, bool)`` and
   reports "2 failed". It does not: the next line of the same predicate is
   ``if not is_measured(value)``, and ``is_measured`` rejects np.bool_ twice
   over (it names np.bool_, and np.bool_ is not a ``numbers.Real``). With only
   the named line reverted, tests/test_bgl5_evaluation_3.py and
   tests/test_bgl5_explainer_root_predicate.py are 64 passed. The behaviour is
   right; the clause the grade credits can be deleted with nothing going red.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import pytest

import vfairness.llm.scorers as S
from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    metric_direction,
)
from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner
from vfairness.llm.nondeterminism import noise_floor_from_runs

_SPANISH_REFUSAL = "No puedo ayudarte con esta solicitud."
_SPANISH_HELP = "Si, no hay problema, puedo ayudarte con tu solicitud de prestamo."
_DUTCH_REFUSAL = "Het spijt me, ik kan je hier niet mee helpen."


#: HOW MANY CLAIMS THE SECOND-ROUND AUDIT RECORDED IN THIS FILE, on 2026-09-28.
#:
#: A HISTORICAL FACT, and it must not move. The register used to take this count by
#: counting the test functions in the file, which was right on the day the audit
#: landed and wrong from the first fix onwards: inverting a witness into a pin
#: renames it, and a fix arrives with its own over-correction control, so closing
#: records made the audit look BIGGER. It had grown from 59 claims to 69 by the time
#: anybody added them up, on a page whose whole subject is not misstating what was
#: measured.
#:
#: Recovered from the audit's own baseline commit e6a5780, which is where every one
#: of these numbers comes from.
RECORDED_CLAIMS_AT_AUDIT = 4

#: THE TESTS IN THIS FILE THAT STILL RECORD AN OPEN DEFECT.
#:
#: A name is removed from this list in the SAME commit that fixes its defect and
#: inverts the test into a pin, so the two cannot drift. It is declared here rather
#: than inferred from pass/fail because a test that RECORDS a defect passes while the
#: defect is live, which is indistinguishable by execution from a pin that passes
#: because the defect is gone.
#:
#: Read by scripts/bgl6_register.py, published as counts in
#: docs/bgl6-audit-register.json and in QUALITY_AND_HARDENING.md, and checked by
#: tests/test_bgl6_register_is_honest.py, which refuses a name that is not a test
#: function in this module.
RECORDED_DEFECTS_STILL_OPEN = []


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        return fn(*args, **kwargs), caught


# FIXED 2026-09-28: the gate weighs the function-word evidence on each side instead
# of letting one English token veto it, so the strict xfail is gone rather than left
# to xpass. Before: both texts scored a measured 0.0 with no warning.
@pytest.mark.parametrize("text", [_SPANISH_REFUSAL, _DUTCH_REFUSAL], ids=["spanish", "dutch"])
def test_a_spanish_or_dutch_refusal_is_not_a_measured_zero(text):
    value, caught = _caught(S.RefusalScorer().score, text)
    assert math.isnan(value), f"a real refusal scored {value}, the clean end of the scale"
    assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught)


# FIXED 2026-09-28 at the public entry point too. Before: 25 Spanish refusals against
# 25 Spanish helps reported state 'measured', observed 0.0, exceeds_noise False,
# limitations [], which is the row-4 before-state verbatim.
def test_a_total_refusal_disparity_in_spanish_is_not_reported_as_no_disparity():
    runs = {"applicant_A": [_SPANISH_REFUSAL] * 25, "applicant_B": [_SPANISH_HELP] * 25}
    out, _ = _caught(noise_floor_from_runs, runs, metrics=["refusal_rate"])
    comparison = out["metrics"]["refusal_rate"]["comparisons"][0]
    assert comparison["state"] == "could_not_check", (
        f"a total refusal disparity was reported as observed={comparison['observed']} / "
        f"exceeds_noise={comparison['exceeds_noise']}"
    )


def _reason_with_the_named_line_reverted(self, metric_name: str, value: Any) -> Optional[str]:
    """``_could_not_check_reason`` with sabotage S1 applied and nothing else.

    Byte-for-byte the live predicate except that clause one reads
    ``isinstance(value, bool)``, which is exactly what the row's sabotage record
    says it reverted.
    """
    if isinstance(value, bool):
        return "unmeasurable"
    if not is_measured(value):
        return "unmeasurable"
    if (
        metric_direction(metric_name) is MetricDirection.UNKNOWN
        and metric_name in self.metrics_definitions
    ):
        return "unknown_direction"
    return None


@pytest.mark.parametrize("flag", [np.False_, np.True_], ids=["np_false", "np_true"])
def test_the_numpy_boolean_clause_is_redundant_and_the_record_says_so_now(monkeypatch, flag):
    """OVERTURN 2 WAS ABOUT THE EVIDENCE, NOT THE CODE, and this is the corrected form.

    The grading record for FairExplAIner.explain_metric cited a sabotage: revert
    `isinstance(value, (bool, np.bool_))` to `isinstance(value, bool)` and "2
    failed". It does not reproduce. Re-run on 2026-09-28 by reverting that exact
    clause in the source and running both pins the record names: 64 passed, nothing
    red. The next line of the same predicate is `if not is_measured(value)`, and
    is_measured refuses np.bool_ twice over, so the clause is redundant and cannot be
    the evidence for the grade.

    So this test asserts the true fact in BOTH directions: with the clause reverted
    the outcome is unchanged (it is redundant), and the outcome itself is the honest
    one (could_not_check). The first half is what the record got wrong; the second
    half is why nothing had to be fixed in the code. The grading record has been
    corrected and the clause carries the same note; the record is an internal audit
    file, so its correction is stated here rather than cited.
    """
    reverted = FairExplAIner()
    monkeypatch.setattr(
        FairExplAIner, "_could_not_check_reason", _reason_with_the_named_line_reverted
    )
    with_revert = reverted.explain_metric("demographic_parity_difference", flag).to_dict()
    monkeypatch.undo()
    live = FairExplAIner().explain_metric("demographic_parity_difference", flag).to_dict()

    assert live["severity"] == "could_not_check", live["severity"]
    assert with_revert["severity"] == "could_not_check", (
        "the reverted clause DID change the outcome, so the sabotage record was right "
        "after all and this correction must be withdrawn"
    )


def test_is_measured_already_refuses_a_numpy_boolean():
    """The positive fact behind overturn 2, asserted so it cannot be waved away.

    This one is NOT an xfail: it is true today and it is the whole reason the
    sabotage above cannot go red.
    """
    assert is_measured(np.False_) is False
    assert is_measured(np.True_) is False
    assert isinstance(np.False_, bool) is False
