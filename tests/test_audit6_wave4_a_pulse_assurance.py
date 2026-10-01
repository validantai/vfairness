"""Audit 6, wave 4, lane A: pulse temporal drift + assurance verdict.

Two findings, each pinned with a REFUSAL pin (the defect, reinstated,
turns this file red) and an OVER-CORRECTION control (the measured path
still returns its real, exact answer).

S-04 ``operations/pulse/agent_probe.py`` ``_temporal_section``
    The drift finding's prose read the Mann-Kendall p-value as
    ``float(feedback.get('p_value') or 1.0)``. A p-value of exactly 0.0
    is the MOST significant reading on the scale and is falsy, so it was
    printed as 1.0, the LEAST significant reading, inside a finding whose
    whole message is that a disparity is growing. The section payload and
    the structured evidence beside it already used the ``is not None``
    form, so the prose contradicted its own evidence field.

S-10 ``operations/reporting/compliance.py`` ``build_assurance_verdict``
    ``recommended`` was in the signature, real callers
    (``orchestrator._assurance`` at both call sites) passed the real
    ``recommend_fairness_definition`` result, and the body never read it:
    the verdict was byte-identical for ``None``, for the real
    recommendation, and for a fabricated one. It is now consumed by the
    disparate-impact recommendation, which previously said "the
    recommended fairness definition" without ever saying which.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vfairness.agents.temporal import TemporalTracker
from vfairness.operations.pulse.agent_probe import (
    _MAX_TEMPORAL_WINDOWS,
    _temporal_section,
)
from vfairness.operations.pulse.recommend import recommend_fairness_definition
from vfairness.operations.reporting import build_assurance_verdict

# The real caller passes (action_col, group_col, deleg_col, "trace_id"); the
# ordering column must NOT be excluded or _find_column skips it.
_EXCLUDE = ("tool", "group", None, "trace_id")


def _drifting_frame(n_windows: int = 10, per_window: int = 20) -> pd.DataFrame:
    """A deterministic, noise-free trace whose 'escalate' rate for group A
    climbs monotonically while group B's stays flat. No RNG, so every
    number this file asserts is reproducible byte for byte."""
    rows = []
    for w in range(n_windows):
        a_hits = int(round((0.05 + 0.10 * w) * per_window))
        for i in range(per_window):
            rows.append(
                {
                    "ts": w * 1000 + i,
                    "group": "A",
                    "tool": "escalate" if i < a_hits else "answer",
                }
            )
        for i in range(per_window):
            rows.append({"ts": w * 1000 + 500 + i, "group": "B", "tool": "answer"})
    return pd.DataFrame(rows)


def _run_temporal(df: pd.DataFrame):
    return _temporal_section(df, df["group"].astype(str), ["A", "B"], "tool", _EXCLUDE)


# ---------------------------------------------------------------------------
# S-04
# ---------------------------------------------------------------------------


class TestATemporalDriftPValueOfZeroIsNotPrintedAsOne:
    """A p-value of 0.0 must reach the reader as 0, never as 1."""

    def test_scipy_really_does_return_exactly_zero(self):
        """The premise, measured rather than assumed: kendalltau underflows
        to exactly 0.0 on a long perfectly monotone series, and 0.0 is
        falsy, which is what the old ``or 1.0`` coalesce keyed on."""
        tau_200, p_200 = stats.kendalltau(np.arange(200), np.arange(200, dtype=float))
        assert float(tau_200) == 1.0
        assert float(p_200) == 0.0
        assert bool(p_200) is False
        # and it is NOT an artefact of "any monotone series": at n=100 the
        # same test returns a tiny but truthy p that always printed fine.
        _, p_100 = stats.kendalltau(np.arange(100), np.arange(100, dtype=float))
        assert float(p_100) == pytest.approx(2.143020576250934e-158, rel=1e-9)
        assert bool(p_100) is True

    def test_zero_p_value_reaches_the_prose_as_zero(self, monkeypatch):
        """REFUSAL PIN.

        ``_temporal_section`` caps itself at ``_MAX_TEMPORAL_WINDOWS`` = 10
        turns, and the exact Kendall test bottoms out at 5.5e-07 for 10
        points (pinned in the control below), so a p of exactly 0.0 cannot
        be FORMED inside this function today. The measurement handed to the
        prose builder here is therefore a real scipy result computed over
        200 real points, injected at the one seam the function uses
        (``TemporalTracker.detect_feedback_loop``). Everything downstream of
        that seam (the drift/growing gate, the severity, the prose, the
        structured evidence) is the real code path.
        """
        tau, p = stats.kendalltau(np.arange(200), np.arange(200, dtype=float))
        assert float(p) == 0.0, "premise: this measurement must be exactly zero"
        measured = {
            "has_feedback_loop": True,
            "trend_direction": "increasing",
            "trend_strength": float(tau),
            "p_value": float(p),
        }
        monkeypatch.setattr(
            TemporalTracker,
            "detect_feedback_loop",
            lambda self, alpha=0.05: dict(measured),
        )

        section, findings = _run_temporal(_drifting_frame())

        assert section["feedbackLoop"]["pValue"] == 0.0
        assert len(findings) == 1, findings
        f = findings[0]
        assert f["type"] == "agent_temporal_drift"
        # The evidence field and the prose must agree, and both must say 0.
        assert f["statisticalTest"]["pValue"] == 0.0
        assert "p 0;" in f["plain"], f["plain"]
        assert "p 1;" not in f["plain"], f["plain"]

    def test_the_measured_path_still_prints_its_real_p_value(self):
        """OVER-CORRECTION CONTROL, exact measured values only.

        No patching: the real TemporalTracker measures a real 10-window
        trace. If the fix had swallowed, rounded or defaulted the p-value,
        these exact numbers would move.
        """
        df = _drifting_frame()
        section, findings = _run_temporal(df)

        assert section["available"] is True
        assert section["windows"] == 10 == _MAX_TEMPORAL_WINDOWS
        assert section["driftDetected"] is True
        fb = section["feedbackLoop"]
        assert fb["hasFeedbackLoop"] is True
        assert fb["trendDirection"] == "increasing"
        assert fb["trendStrength"] == pytest.approx(0.9999999999999999, rel=1e-12)
        # The exact floor of the exact Kendall test at 10 points. This is
        # also the proof that p == 0.0 is unreachable inside this function
        # at the current window cap.
        assert fb["pValue"] == pytest.approx(5.511463844797178e-07, rel=1e-9)

        assert len(findings) == 1, findings
        f = findings[0]
        assert f["severity"] == "critical"
        assert f["attribute"] == "ts"
        assert f["statisticalTest"]["pValue"] == pytest.approx(5.511463844797178e-07, rel=1e-9)
        assert f["statisticalTest"]["windows"] == 10
        assert "p 5.511e-07;" in f["plain"], f["plain"]

    def test_a_mid_range_p_value_is_printed_unchanged(self):
        """OVER-CORRECTION CONTROL 2: an ordinary, clearly non-zero p-value
        survives the same code path with its exact value."""
        tau, p = stats.kendalltau(np.arange(10), np.arange(10, dtype=float))
        measured_p = 0.0123456789
        monkeypatch_target = {
            "has_feedback_loop": True,
            "trend_direction": "increasing",
            "trend_strength": float(tau),
            "p_value": measured_p,
        }
        original = TemporalTracker.detect_feedback_loop
        try:
            TemporalTracker.detect_feedback_loop = (  # type: ignore[method-assign]
                lambda self, alpha=0.05: dict(monkeypatch_target)
            )
            _, findings = _run_temporal(_drifting_frame())
        finally:
            TemporalTracker.detect_feedback_loop = original  # type: ignore[method-assign]

        assert len(findings) == 1, findings
        assert findings[0]["statisticalTest"]["pValue"] == measured_p
        assert "p 0.01235;" in findings[0]["plain"], findings[0]["plain"]


# ---------------------------------------------------------------------------
# S-10
# ---------------------------------------------------------------------------

_DISPARATE_ROW = {
    "assessable": True,
    "attribute": "race",
    "worstGroup": "Black",
    "referenceGroup": "White",
    "gap": 0.23,
    "significant": True,
}

_GENERIC_ACTION = (
    "Apply the recommended fairness definition and a mitigation "
    "(reweighing / threshold optimisation), then re-measure."
)


def _mitigation_action(**kwargs) -> str:
    """The one disparate-impact recommendation the real verdict emits."""
    verdict = build_assurance_verdict(
        per_variable=[dict(_DISPARATE_ROW)],
        domain=kwargs.pop("domain", "hiring"),
        jurisdiction=kwargs.pop("jurisdiction", "EU"),
        has_truth=True,
        **kwargs,
    )
    hits = [
        r for r in verdict["recommendations"] if r["vfairnessFunction"] == "recommend_interventions"
    ]
    assert len(hits) == 1, verdict["recommendations"]
    return hits[0]["action"]


class TestTheAssuranceVerdictActuallyConsumesTheRecommendation:
    def test_the_recommendation_is_named_in_the_mitigation_sentence(self):
        """REFUSAL PIN.

        The sentence says "the recommended fairness definition"; when a real
        recommendation is handed in, it must say WHICH one. If ``recommended``
        goes inert again the sentence collapses back to the generic wording
        and this fails.
        """
        rec = recommend_fairness_definition("hiring", "EU", True)
        assert rec["primary"] == "Equal opportunity"
        action = _mitigation_action(recommended=rec)
        assert "(Equal opportunity)" in action, action
        assert action != _GENERIC_ACTION

    def test_a_different_recommendation_produces_a_different_sentence(self):
        """REFUSAL PIN 2: the value is read, not merely present. Two real
        recommendations with different primaries must not collapse onto one
        sentence, which is exactly what an ignored parameter does."""
        hiring = recommend_fairness_definition("hiring", "EU", True)
        recidivism = recommend_fairness_definition("recidivism", "US", True)
        assert hiring["primary"] == "Equal opportunity"
        assert recidivism["primary"] == "Equalized odds"

        a = _mitigation_action(recommended=hiring)
        b = _mitigation_action(recommended=recidivism, domain="recidivism", jurisdiction="US")
        assert "(Equal opportunity)" in a, a
        assert "(Equalized odds)" in b, b
        assert a != b

    @pytest.mark.parametrize(
        "recommended",
        [
            pytest.param(None, id="None"),
            pytest.param({}, id="empty-dict-what-the-agent-probes-pass"),
            pytest.param({"primary": None}, id="primary-present-holding-None"),
            pytest.param({"primary": ""}, id="primary-empty"),
            pytest.param({"primary": "   "}, id="primary-blank"),
            pytest.param({"primary": 12}, id="primary-not-a-string"),
            pytest.param("Equal opportunity", id="not-a-dict-at-all"),
        ],
    )
    def test_no_recommendation_means_no_named_definition(self, recommended):
        """OVER-CORRECTION CONTROL: the sentence must never name a definition
        it was not handed. ``agent_probe``, ``llm_probe`` and ``vision_probe``
        all pass ``{}`` on purpose, so the generic wording is the correct,
        exact output there, not a placeholder, and never an invented name."""
        assert _mitigation_action(recommended=recommended) == _GENERIC_ACTION

    def test_the_recommendation_reroutes_nothing_else_in_the_verdict(self):
        """OVER-CORRECTION CONTROL: honouring ``recommended`` must change the
        one sentence and nothing else. Exact measured verdict values, not
        membership of a broad set."""
        rec = recommend_fairness_definition("hiring", "EU", True)
        without = build_assurance_verdict(
            per_variable=[dict(_DISPARATE_ROW)], domain="hiring", jurisdiction="EU", has_truth=True
        )
        with_rec = build_assurance_verdict(
            per_variable=[dict(_DISPARATE_ROW)],
            domain="hiring",
            jurisdiction="EU",
            has_truth=True,
            recommended=rec,
        )

        assert without["overall"] == "Adverse"
        assert with_rec["overall"] == "Adverse"
        assert without["blocksDeployment"] is True
        assert with_rec["blocksDeployment"] is True
        assert with_rec["oneLineVerdict"] == without["oneLineVerdict"]
        assert [(f["id"], f["type"], f["severity"]) for f in with_rec["findings"]] == [
            ("DSP-001", "disparate_impact", "critical")
        ]
        assert [(f["id"], f["type"], f["severity"]) for f in with_rec["findings"]] == [
            (f["id"], f["type"], f["severity"]) for f in without["findings"]
        ]
        assert [(r["id"], r["priority"], r["routeTo"]) for r in with_rec["recommendations"]] == [
            (r["id"], r["priority"], r["routeTo"]) for r in without["recommendations"]
        ]
        differing = [
            (a["id"], a["action"], b["action"])
            for a, b in zip(with_rec["recommendations"], without["recommendations"])
            if a["action"] != b["action"]
        ]
        assert len(differing) == 1, differing
        assert differing[0][0] == "REC-001"
