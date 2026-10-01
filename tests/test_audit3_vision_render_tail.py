"""Third-iteration audit fixes: vision-render-tail unit.

Pins three findings from docs/audits/third-iteration-audit-2026-08-22.md:

1. [HIGH] vfairness.vision.representation_severity was blind to
   under-representation / exclusion (keyed on MaxSkew only), so a severely
   under-represented or excluded group passed as "Within representation
   tolerance". Now gates on the WORST absolute skew, max(|MaxSkew|,|MinSkew|).
2. [LOW] xai.diagnostics.local_r_squared returned 1.0 for a degenerate
   constant-target sample even against an arbitrary surrogate. Now matches
   sklearn.metrics.r2_score (0.0 unless the residual is ~0).
3. [LOW] explainer training-report DP card printed an interpretation guide
   (0.05/0.10) that contradicted its own severity cutoffs (0.08/0.15). The
   guide now mirrors the cutoffs, so it cannot call an info-severity value
   "borderline".

Each finding: the defect case FIRST (the scenario that used to give the wrong
answer now gives the right one), then a does-not-overcorrect case.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from vfairness import explainer
from vfairness.operations.pulse.vision_probe import vision_probe_pulse
from vfairness.vision import representation_severity, skew
from vfairness.xai.diagnostics.faithfulness import local_r_squared

# ---------------------------------------------------------------------------
# Finding 1 (HIGH): representation gate must catch under-representation
# ---------------------------------------------------------------------------


def test_representation_severity_flags_severe_underrepresentation():
    # DEFECT CASE: group A at 1/99 -> MaxSkew ~0.19 (small), MinSkew ~ -2.80.
    # Keyed on MaxSkew alone this returned "pass".
    labels = ["A"] * 1 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20 + ["E"] * 20 + ["F"] * 18
    s = skew(labels)
    assert s["maxSkew"] < 0.22, "precondition: MaxSkew alone would have said pass"
    assert s["minSkew"] <= -0.5, "precondition: MinSkew marks the severe under-representation"
    # Full dict form (the fixed call path):
    assert representation_severity(s) == "critical"
    # Explicit two-arg form:
    assert representation_severity(s["maxSkew"], s["minSkew"]) == "critical"


def test_representation_severity_flags_total_exclusion():
    # DEFECT CASE: a whole group excluded from the set vs a uniform reference.
    ref = {g: 1.0 / 6 for g in "ABCDEF"}
    labels = ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20 + ["E"] * 20  # no F
    s = skew(labels, ref)

    # READINESS-6, 2026-09-10. This asserted `s["minSkew"] < -1.0`, which was
    # true only because `skew` clamped the absent group's observed share to 1e-9
    # and took its logarithm: a magnitude decided entirely by the epsilon (at
    # 1e-6 the same absence scored -12.4 instead of -19.3). Removing that
    # fabrication also removed the route by which total exclusion reached the
    # gate, and this test went red, correctly. The SUBJECT of the test is the
    # last line and it is unchanged; only the mechanism it asserts has moved
    # from an invented number to the named fact.
    assert s["perGroup"]["F"] is None, "an absent group must not carry a skew value"
    assert s["absentGroups"] == ["F"]
    assert s["maxSkew"] < 0.22, "the groups that ARE present are only mildly skewed"
    assert representation_severity(s) == "critical"


def test_vision_probe_emits_finding_for_underrepresentation_end_to_end():
    # DEFECT CASE end-to-end: no reference supplied, so the amplification path
    # is inert and the representation gate is the only detector. Before the fix
    # this produced 0 findings and "Within representation tolerance".
    labels = ["A"] * 1 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20 + ["E"] * 20 + ["F"] * 18
    df = pd.DataFrame({"detected_race": labels})
    out = vision_probe_pulse(df, {}, "detected_race", "generic", "EU")
    types = [f["type"] for f in out["data"]["bias"]]
    assert "image_representation_skew" in types
    rep = next(f for f in out["data"]["bias"] if f["type"] == "image_representation_skew")
    assert rep["severity"] in ("warn", "critical")
    assert "finding" in out["data"]["vision"]["summary"]


def test_representation_severity_does_not_overcorrect_on_balanced_set():
    # NO-OVERCORRECT: a balanced set is still pass, and the probe emits no
    # representation-skew finding.
    labels = ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20
    s = skew(labels)
    assert representation_severity(s) == "pass"
    df = pd.DataFrame({"detected_race": labels})
    out = vision_probe_pulse(df, {}, "detected_race", "generic", "EU")
    types = [f["type"] for f in out["data"]["bias"]]
    assert "image_representation_skew" not in types
    assert "Within representation tolerance" in out["data"]["vision"]["summary"]


def test_representation_severity_preserves_float_contract():
    # NO-OVERCORRECT: the legacy single-float API (pinned by tests/unit/
    # test_vision.py) is unchanged; a bare float thresholds on its magnitude.
    assert representation_severity(0.1) == "pass"
    assert representation_severity(0.3) == "warn"
    assert representation_severity(0.6) == "critical"
    assert representation_severity(-0.6) == "critical"


# ---------------------------------------------------------------------------
# Finding 2 (LOW): local_r_squared on a degenerate constant target
# ---------------------------------------------------------------------------


def test_local_r_squared_constant_target_matches_sklearn():
    # DEFECT CASE: constant true output, arbitrary surrogate -> used to be 1.0.
    sur = np.array([0.1, 0.9, 0.2, 0.7])
    true_const = np.array([0.5, 0.5, 0.5, 0.5])
    got = local_r_squared(surrogate_predictions=sur, true_predictions=true_const)
    assert got == pytest.approx(r2_score(true_const, sur))  # sklearn -> 0.0
    assert got == pytest.approx(0.0)


def test_local_r_squared_perfect_constant_fit_still_one():
    # NO-OVERCORRECT: a perfect (zero-residual) fit on a constant target is 1.0.
    true_const = np.array([0.5, 0.5, 0.5, 0.5])
    got = local_r_squared(surrogate_predictions=true_const.copy(), true_predictions=true_const)
    assert got == pytest.approx(1.0)


def test_local_r_squared_nondegenerate_unchanged():
    # NO-OVERCORRECT: a normal, non-constant sample still matches sklearn.
    true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sur = np.array([0.11, 0.19, 0.31, 0.39, 0.52])
    got = local_r_squared(surrogate_predictions=sur, true_predictions=true)
    assert got == pytest.approx(r2_score(true, sur))


# ---------------------------------------------------------------------------
# Finding 3 (LOW): DP-card guide must agree with its severity cutoffs
# ---------------------------------------------------------------------------


def _baseline_card(dp: float):
    rep = SimpleNamespace(
        recommendation=None,
        method_comparisons=[],
        baseline_metrics={"demographic_parity_difference": dp},
        action_items=[],
        task_type="classification",
    )
    out = explainer._explain_training_report(rep)
    card = out.explanations[0]
    assert card.metric_name == "Baseline Fairness Assessment"
    return card


def test_dp_card_guide_does_not_contradict_severity_in_info_band():
    # DEFECT CASE: DP = 0.06 is severity=info / "reasonably fair", yet the old
    # guide banded 0.05-0.10 as "borderline" -- a direct contradiction. The
    # guide's fair ceiling must now cover the whole info-severity range.
    card = _baseline_card(0.06)
    assert card.severity == "info"
    assert "reasonably fair" in card.evaluation
    guide = card.interpretation_guide
    # The guide must key off the real cutoffs (0.08 / 0.15), not the old
    # contradicting 0.05 / 0.10.
    assert "0.08" in guide and "0.15" in guide
    assert "0.05" not in guide and "0.10" not in guide
    # 0.06 is inside the guide's "reasonably fair" band, not "borderline".
    assert "<= 0.08: reasonably fair" in guide


def test_dp_card_severity_cutoffs_unchanged():
    # NO-OVERCORRECT: the severity/prose contract (wave5) is untouched -- the
    # fix only realigned the printed guide, never the cutoffs.
    assert _baseline_card(0.08).severity == "info"
    assert _baseline_card(0.0801).severity == "medium"
    assert _baseline_card(0.15).severity == "medium"
    assert _baseline_card(0.1501).severity == "high"
    # And the guide's high-band boundary matches the high-severity cutoff.
    assert "> 0.15: likely unfair" in _baseline_card(0.2).interpretation_guide


def test_a_mapping_missing_one_skew_is_not_a_pass():
    """READINESS-5, 2026-09-10. ONE key missing substituted 0.0 for it and
    banded on the other, in silence.

    Measured that day::

        representation_severity({"maxSkew": 0.10, "minSkew": -0.90}) -> 'critical'
        representation_severity({"maxSkew": 0.10})                   -> 'pass'

    This function's own docstring says gating on MaxSkew alone "is blind to a
    group that is severely under-represented or excluded (a large negative
    MinSkew) -- the dominant representation harm", and that is exactly what it
    did whenever MinSkew was absent. Two sibling absence cases already warned
    and returned 'not_assessed'; this third one returned a band and said
    nothing.

    The asymmetry ran the dangerous way: a missing MAXskew still yields
    'critical', because 0.0 cannot hide a large |minSkew|, so only the
    reassuring direction was ever reachable.
    """
    import warnings

    for label, payload in (
        ("key absent", {"maxSkew": 0.10}),
        ("key present but None", {"maxSkew": 0.10, "minSkew": None}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = representation_severity(payload)
        assert out == "not_assessed", f"{label}: banded {out!r} on one skew of two"
        assert any("not 'pass'" in str(w.message) for w in caught), (
            f"{label}: the withheld band was not explained"
        )


def test_a_partial_mapping_still_reports_a_band_it_can_already_justify():
    """OVER-CORRECTION CONTROL, and the reason the fix is not a blanket refusal.

    `max(|maxSkew|, |minSkew|)` is MONOTONE in the missing value, so a band
    computed from the skew we DO have is a lower bound: more evidence can only
    raise it. Withholding a 'critical' that one measured skew already justifies
    would lose a real finding to protect against a hypothetical one.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert representation_severity({"maxSkew": 0.80}) == "critical"
        assert representation_severity({"minSkew": -0.90}) == "critical"
        assert representation_severity({"maxSkew": 0.30}) == "warn"
    assert not caught, "a band that stands on its own evidence was warned about"

    # And the documented bare-float path keeps its magnitude-only behaviour,
    # which is an explicit backward-compatibility promise in the docstring.
    assert representation_severity(0.10) == "pass"
    assert representation_severity(0.80) == "critical"

    # Both keys present is untouched in both directions.
    assert representation_severity({"maxSkew": 0.10, "minSkew": -0.90}) == "critical"
    assert representation_severity({"maxSkew": 0.10, "minSkew": -0.05}) == "pass"
