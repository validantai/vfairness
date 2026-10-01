"""Pulse agent probe: a NaN effect size must not be graded "info".

``ActionBiasAnalyzer._cohens_d`` was changed on 2026-09-08 to return
``nan`` when the pooled standard deviation is zero, because returning 0.0
there ("no effect") for two constant, SEPARATED groups was the exact
inverse of the truth. The producer was fixed; the function that GRADES
its result was not.

``agent_probe._severity_from_d`` tested ``abs(d) >= 0.8`` then
``abs(d) >= 0.5``. Both comparisons are False against ``nan``, so the
function fell through to ``return "info"``, rank 0 in the assurance
grader's ``_SEV_RANK``, the weakest word the scale has. Its only call
site, ``_trajectory_section``, passed the value straight in.

Measured on the fixture below, before the fix:

    comparison: meanA 10.00, meanB 100.00, effectSize nan,
                pValue 0.000999, pAdjusted 0.000999, significant True
    finding   : agent_trajectory_bias, severity 'info',
                plain "... Cohen's d nan ..."
    assurance : Disclaimer, blocksDeployment False, tone 'disclaimer',
                "Insufficient assessable data to issue a fairness opinion."

Every episode of one group took 10 steps and every episode of the other
took 100. That is perfect separation, the strongest gap the scale can
describe, at p = 0.000999 after Benjamini-Hochberg correction, and it was
published as the weakest state the product has.

Every test here is either a REFUSAL PIN (an unmeasured effect size must
never read as a measured small one) or an OVER CORRECTION CONTROL (a
MEASURED effect size must still grade exactly as it did before, in every
band, and a measured zero must not be promoted to perfect separation).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.agents import ActionBiasAnalyzer
from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.agent_probe import _severity_from_d, agent_probe_pulse

SEED = 7


# ── fixtures ────────────────────────────────────────────────────────────────


def _traces(steps, n_per_group=30):
    """Per-episode agent traces: two adequately sampled groups (the probe's
    floor is 20 episodes), an identical tool mix on both sides so the tool
    family finds nothing, and the supplied step counts.

    ``dtype=object`` on the labels deliberately: a numpy "<U5" array
    truncates the longer group name and voids the fixture.
    """
    groups = np.array(["group_alpha"] * n_per_group + ["group_beta"] * n_per_group, dtype=object)
    tools = (["search", "email", "escalate"] * (n_per_group // 3)) * 2
    return pd.DataFrame({"group": groups, "tool": tools, "steps": list(steps)})


def _separated_traces(n_per_group=30):
    """Perfect separation: EVERY episode of one group took 10 steps and
    every episode of the other took 100. Zero variance inside each group,
    so the pooled standard deviation is 0 and Cohen's d is undefined."""
    return _traces([10.0] * n_per_group + [100.0] * n_per_group, n_per_group)


def _noisy_traces(mean_a, mean_b, sd=1.0, n_per_group=60):
    """A real, noisy step-count gap: the effect size IS computable here."""
    rng = np.random.default_rng(SEED)
    a = [float(v) for v in rng.normal(mean_a, sd, n_per_group)]
    b = [float(v) for v in rng.normal(mean_b, sd, n_per_group)]
    return _traces(a + b, n_per_group)


def _probe(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = agent_probe_pulse(df, {}, "tool", "group", "hiring", "EU")
    assert out["success"] is True
    return out["data"]


def _trajectory_comparison(data):
    metrics = [m for m in data["agent"]["trajectory"]["perMetric"] if m["metric"] == "steps"]
    assert metrics, "the fixture carries a step column, so the metric must be present"
    comparisons = metrics[0]["comparisons"]
    assert len(comparisons) == 1
    return comparisons[0]


def _trajectory_findings(data):
    return [f for f in data["bias"] if f["type"] == "agent_trajectory_bias"]


# ── refusal pins ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("unmeasured", [float("nan"), float("inf"), float("-inf"), None])
def test_severity_from_d_refuses_an_unmeasured_effect_size(unmeasured):
    """REFUSAL PIN. The grader must report "no severity was measured",
    never the weakest word on the scale. ``abs(nan) >= 0.8`` is False and
    so is ``abs(nan) >= 0.5``, which is exactly how a nan reached
    ``return "info"``."""
    assert _severity_from_d(unmeasured) is None
    assert _severity_from_d(unmeasured) != "info"


def test_perfect_separation_is_graded_critical_not_info():
    """REFUSAL PIN. Both groups constant and 90 steps apart: Cohen's d is
    undefined because its denominator is zero, and the permutation
    p-value beside it IS measured and significant. The finding must be
    graded off what was measured."""
    data = _probe(_separated_traces())
    comp = _trajectory_comparison(data)
    assert comp["meanA"] == 10.0
    assert comp["meanB"] == 100.0
    assert comp["significant"] is True
    assert comp["pAdjusted"] < 0.05
    # Three states: the effect size is reported as unmeasured, and the
    # reason it is unmeasured is stated, not left for the reader to guess.
    assert comp["effectSize"] is None
    assert comp["effectSizeMeasured"] is False
    assert comp["perfectSeparation"] is True

    findings = _trajectory_findings(data)
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert findings[0]["severity"] != "info"


def test_perfect_separation_blocks_deployment():
    """REFUSAL PIN, at the surface a reader actually sees. Before the fix
    this exact frame produced "Disclaimer / Insufficient assessable data
    to issue a fairness opinion", blocksDeployment False, tone
    'disclaimer'."""
    data = run_pulse(
        _separated_traces(), {"source_kind": "agent", "protected_attributes": ["group"]}
    )["data"]
    assert data["sourceKind"] == "agent_traces"
    assurance = data["assurance"]
    assert assurance["overall"] == "Adverse"
    assert assurance["blocksDeployment"] is True
    assert data["verdict"]["tone"] == "critical"
    assert "Insufficient assessable data" not in assurance["oneLineVerdict"]


def test_undefined_effect_size_is_never_formatted_as_a_number():
    """REFUSAL PIN. ``f"Cohen's d {nan:.2f}"`` rendered the string
    "Cohen's d nan" into the plain-language finding and into the assurance
    one-liner: an unmeasured value printed where a measurement goes."""
    data = _probe(_separated_traces())
    plain = _trajectory_findings(data)[0]["plain"]
    assert "nan" not in plain.lower()
    assert "perfectly separated" in plain
    test = _trajectory_findings(data)[0]["statisticalTest"]
    assert test["cohensD"] is None
    assert test["cohensDMeasured"] is False
    assert test["perfectSeparation"] is True


def test_unmeasured_effect_size_serialises_as_null_never_as_nan():
    """REFUSAL PIN. A raw nan on the contract serialises as the ``NaN``
    token, which strict JSON parsers reject, and reads as a number to
    anything that does accept it."""
    data = _probe(_separated_traces())
    payload = json.dumps(data, allow_nan=False)
    assert '"effectSize": null' in payload
    assert "NaN" not in payload


def test_effect_size_that_could_not_be_computed_is_could_not_check(monkeypatch):
    """REFUSAL PIN, third state. When the effect size is unavailable for a
    reason that is NOT perfect separation, the finding must say so rather
    than borrow a severity. The old fallback default at this call site was
    ``0.0``, a measurement meaning "no effect", which then graded "info"."""

    def _unavailable(group_a, group_b):
        raise RuntimeError("effect size unavailable")

    monkeypatch.setattr(ActionBiasAnalyzer, "_cohens_d", staticmethod(_unavailable))
    data = _probe(_noisy_traces(5.0, 10.0))

    comp = _trajectory_comparison(data)
    assert comp["effectSize"] is None
    assert comp["effectSizeMeasured"] is False
    assert comp["perfectSeparation"] is False
    assert comp["significant"] is True

    findings = _trajectory_findings(data)
    assert len(findings) == 1
    assert findings[0]["severity"] == "insufficient_data"
    assert findings[0]["severity"] not in ("info", "warn", "critical")
    assert "could not be computed" in findings[0]["plain"]

    # The assurance grader ranks that word None, so the finding is carried
    # as explicitly unassessed and never as a clean or a minor result.
    unassessed = [f for f in data["assurance"]["findings"] if f.get("assessed") is False]
    assert len(unassessed) == 1
    assert unassessed[0]["type"].startswith("agent_trajectory_bias")
    assert unassessed[0]["severity"] == "insufficient_data"


# ── over correction controls ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "d, expected",
    [
        (0.0, "info"),
        (0.2, "info"),
        (-0.49, "info"),
        (0.5, "warn"),
        (-0.79, "warn"),
        (0.8, "critical"),
        (-6.2, "critical"),
    ],
)
def test_measured_effect_sizes_still_grade_by_their_bands(d, expected):
    """OVER CORRECTION CONTROL. Every MEASURED value must grade exactly as
    it did before, including a measured 0.0, which is a real "no effect"
    and must stay "info" rather than becoming a could-not-check."""
    assert _severity_from_d(d) == expected


def test_measured_large_gap_is_still_critical_and_still_prints_its_d():
    """OVER CORRECTION CONTROL, end to end with real numbers: 5 steps vs
    10 steps at sd 1. The effect size IS computable here and must be
    reported as the number it is."""
    data = _probe(_noisy_traces(5.0, 10.0))
    comp = _trajectory_comparison(data)
    assert comp["effectSizeMeasured"] is True
    assert comp["perfectSeparation"] is False
    assert comp["effectSize"] is not None
    assert comp["effectSize"] < -0.8

    findings = _trajectory_findings(data)
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert f"Cohen's d {comp['effectSize']:.2f}" in findings[0]["plain"]
    assert findings[0]["statisticalTest"]["cohensDMeasured"] is True
    assert data["assurance"]["overall"] == "Adverse"


def test_measured_medium_gap_is_still_warn_and_still_qualified():
    """OVER CORRECTION CONTROL. A medium measured effect must not be
    promoted by the new branch."""
    data = _probe(_noisy_traces(5.0, 5.6, n_per_group=120))
    comp = _trajectory_comparison(data)
    assert comp["effectSizeMeasured"] is True
    assert comp["perfectSeparation"] is False
    assert -0.8 < comp["effectSize"] < -0.5

    findings = _trajectory_findings(data)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warn"
    assert data["assurance"]["overall"] == "Qualified"


def test_measured_small_gap_is_still_info():
    """OVER CORRECTION CONTROL. A small measured effect stays at the
    bottom of the scale even though it is statistically significant: the
    fix must not turn significance itself into a severity."""
    data = _probe(_noisy_traces(5.0, 5.35, n_per_group=300))
    comp = _trajectory_comparison(data)
    assert comp["effectSizeMeasured"] is True
    assert comp["perfectSeparation"] is False
    assert -0.5 < comp["effectSize"] < 0.0
    assert comp["significant"] is True

    findings = _trajectory_findings(data)
    assert len(findings) == 1
    assert findings[0]["severity"] == "info"


def test_two_constant_but_equal_groups_are_a_measured_zero():
    """OVER CORRECTION CONTROL. Zero within-group variance alone is NOT
    perfect separation. When both groups are constant AND equal the
    producer returns a real 0.0, and the probe must keep reporting it as a
    measurement rather than escalating it."""
    data = _probe(_traces([7.0] * 60))
    comp = _trajectory_comparison(data)
    assert comp["effectSize"] == 0.0
    assert comp["effectSizeMeasured"] is True
    assert comp["perfectSeparation"] is False
    assert comp["significant"] is False
    assert _trajectory_findings(data) == []
